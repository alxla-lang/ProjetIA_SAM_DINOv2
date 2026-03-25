# ============================================================
# INSTALLATION DES DEPENDANCES
# ============================================================

import os, gc, glob, random, warnings, time, csv
from collections import defaultdict, Counter
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast

import torchvision.transforms as T
import torchvision.models as models

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             roc_curve, auc, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.loader import DataLoader as PyGDataLoader

warnings.filterwarnings('ignore')

# Configurer l'environnement d'exécution (CPU/GPU) et la reproductibilité
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if DEVICE.type == 'cuda':
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

print(f"Device: {DEVICE}")
if DEVICE.type == 'cuda':
    p = torch.cuda.get_device_properties(0)

CLASSES = ['Sky', 'Building', 'Pole', 'Road', 'Sidewalk',
           'Tree', 'Sign', 'Fence', 'Car', 'Pedestrian', 'Bicyclist']
NUM_CLASSES = len(CLASSES)
VOID = -1

CLASS_GROUPING = {
    'Sky': 0, 'Building': 1, 'Archway': 1, 'Bridge': 1, 'Tunnel': 1, 'Wall': 1,
    'Column_Pole': 2, 'TrafficCone': 2,
    'Road': 3, 'LaneMkgsDriv': 3, 'LaneMkgsNonDriv': 3, 'RoadShoulder': 3,
    'Sidewalk': 4, 'ParkingBlock': 4,
    'Tree': 5, 'VegetationMisc': 5, 'Animal': 5,
    'SignSymbol': 6, 'Misc_Text': 6, 'TrafficLight': 6,
    'Fence': 7,
    'Car': 8, 'CartLuggagePram': 8, 'MotorcycleScooter': 8, 'OtherMoving': 8,
    'SUVPickupTruck': 8, 'Train': 8, 'Truck_Bus': 8,
    'Pedestrian': 9, 'Child': 9,
    'Bicyclist': 10,
    'Void': VOID, 'Unlabelled': VOID,
}

MIN_AREA = 500  

# Libérer la mémoire
def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Afficher les informations VRAM
def vram_info():
    if torch.cuda.is_available():
        a = torch.cuda.memory_allocated() / 1e9
        r = torch.cuda.memory_reserved() / 1e9
        return f"Alloc: {a:.2f} GB | Reserve: {r:.2f} GB"
    return "CPU mode"

print(f"Configuration terminee. {NUM_CLASSES} classes semantiques.")

# ============================================================
# TELECHARGEMENT : CamVid + SAM
# ============================================================
import os
import subprocess

if not os.path.exists('CamVid'):
    print("Telechargement de CamVid...")
    subprocess.run(["git", "clone", "https://github.com/lih627/CamVid.git"])

CAMVID = 'CamVid'
print("Contenu:", os.listdir(CAMVID))

COLOR_MAP = {}
csv_path = os.path.join(CAMVID, 'class_dict.csv')
if os.path.exists(csv_path):
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row['name'].strip()
            r, g, b = int(row['r']), int(row['g']), int(row['b'])
            cls = CLASS_GROUPING.get(name, VOID)
            COLOR_MAP[(r, g, b)] = cls
    print(f"Color map depuis CSV: {len(COLOR_MAP)} couleurs")
else:
    COLOR_MAP = {
        (128,128,128):0, (128,0,0):1, (192,0,128):1, (0,128,64):1,
        (64,0,64):1, (64,192,0):1, (192,192,128):2, (0,0,64):2,
        (128,64,128):3, (128,0,192):3, (192,0,64):3, (128,128,192):3,
        (0,0,192):4, (64,192,128):4, (128,128,0):5, (192,192,0):5,
        (64,128,64):5, (192,128,128):6, (128,128,64):6, (0,64,64):6,
        (64,64,128):7, (64,0,128):8, (64,0,192):8, (192,0,192):8,
        (128,64,64):8, (64,128,192):8, (192,64,128):8, (192,128,192):8,
        (64,64,0):9, (192,128,64):9, (0,128,192):10, (0,0,0):VOID,
    }
    print("Color map par defaut.")

splits = {}
for sn in ['train', 'val', 'test']:
    img_d = lbl_d = None
    for d in os.listdir(CAMVID):
        full = os.path.join(CAMVID, d)
        if not os.path.isdir(full): continue
        dl = d.lower()
        if dl == sn: img_d = full
        elif dl in [f'{sn}_labels', f'{sn}annot', f'{sn}label']: lbl_d = full
    if img_d and lbl_d:
        imgs = sorted(glob.glob(os.path.join(img_d, '*.png')))
        pairs = []
        for ip in imgs:
            bn = os.path.basename(ip)
            for lp_cand in [os.path.join(lbl_d, bn),
                            os.path.join(lbl_d, bn.replace('.png', '_L.png'))]:
                if os.path.exists(lp_cand):
                    pairs.append((ip, lp_cand)); break
        splits[sn] = pairs
        print(f"  {sn}: {len(pairs)} paires image-label")

train_pairs = splits.get('train', [])
val_pairs = splits.get('val', [])
test_pairs = splits.get('test', [])

if not train_pairs:
    all_p = []
    for v in splits.values(): all_p.extend(v)
    random.shuffle(all_p); n = len(all_p)
    train_pairs = all_p[:int(0.7*n)]
    val_pairs = all_p[int(0.7*n):int(0.85*n)]
    test_pairs = all_p[int(0.85*n):]

print(f"\nSplits: Train={len(train_pairs)}, Val={len(val_pairs)}, Test={len(test_pairs)}")

SAM_CKPT = 'sam_vit_h_4b8939.pth'
if not os.path.exists(SAM_CKPT):
    print("Telechargement des poids SAM...")
    subprocess.run(["wget", "-q", "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"])

# Convertir un label RGB en carte de classes
def label_to_classmap(label_path):
    img = cv2.imread(label_path)
    if img is None: return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    cmap = np.full((h, w), VOID, dtype=np.int32)
    for color, cls in COLOR_MAP.items():
        mask = np.all(rgb == np.array(color, dtype=np.uint8), axis=-1)
        cmap[mask] = cls
    return cmap

img_d = os.path.join(CAMVID, 'CamVid_RGB')
lbl_d = os.path.join(CAMVID, 'CamVid_Label')

all_pairs = []

if os.path.exists(img_d) and os.path.exists(lbl_d):
    imgs = sorted(glob.glob(os.path.join(img_d, '*.png')))
    for ip in imgs:
        bn = os.path.basename(ip)
        lp_cand_1 = os.path.join(lbl_d, bn.replace('.png', '_L.png'))
        lp_cand_2 = os.path.join(lbl_d, bn) 

        if os.path.exists(lp_cand_1):
            all_pairs.append((ip, lp_cand_1))
        elif os.path.exists(lp_cand_2):
            all_pairs.append((ip, lp_cand_2))

import random
random.shuffle(all_pairs)
n = len(all_pairs)

train_pairs = all_pairs[:int(0.7 * n)]
val_pairs = all_pairs[int(0.7 * n):int(0.85 * n)]
test_pairs = all_pairs[int(0.85 * n):]

print(f"\nSplits: Train={len(train_pairs)}, Val={len(val_pairs)}, Test={len(test_pairs)}")

sample_cmap = label_to_classmap(train_pairs[0][1])
if sample_cmap is not None:
    uniq, cnts = np.unique(sample_cmap, return_counts=True)
    print("\nClasses dans label exemple:")
    for u, c in zip(uniq, cnts):
        name = CLASSES[u] if u >= 0 else 'Void'
        print(f"  {name}: {c} px ({100*c/sample_cmap.size:.1f}%)")

# ============================================================
# MODELES : EXTRACTEUR DE FEATURES + MLP DE GRANULARITE
# ============================================================

class ImageFeatureExtractor(nn.Module):
    # Initialiser l'extracteur de features
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    # Faire passer l'image dans le réseau
    def forward(self, x):
        with torch.no_grad():
            return self.backbone(x).squeeze(-1).squeeze(-1)

class GranularityMLP(nn.Module):
    # Initialiser le MLP de granularité
    def __init__(self, in_dim=512, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden // 2, 3),
        )

    # Faire une prédiction avec le MLP
    def forward(self, features):
        raw = self.net(features)
        pps  = torch.sigmoid(raw[:, 0]) * 28.0 + 4.0    
        iou  = torch.sigmoid(raw[:, 1]) * 0.28 + 0.70   
        stab = torch.sigmoid(raw[:, 2]) * 0.28 + 0.70   
        return torch.stack([pps, iou, stab], dim=-1)

feat_extractor = ImageFeatureExtractor().to(DEVICE)
mlp = GranularityMLP().to(DEVICE)

resnet_transform = T.Compose([
    T.Resize((224, 224)), T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

n_feat = sum(p.numel() for p in feat_extractor.parameters())
n_mlp = sum(p.numel() for p in mlp.parameters() if p.requires_grad)
print(f"Feature Extractor: {n_feat/1e6:.1f}M params (geles)")
print(f"MLP Granularite: {n_mlp:,} params (entrainables)")

# ============================================================
# PRE-CALCUL DES PARAMETRES OPTIMAUX + ENTRAINEMENT MLP
# ============================================================

sam = sam_model_registry["vit_h"](checkpoint=SAM_CKPT)
sam.to(DEVICE).eval()
for p in sam.parameters(): p.requires_grad = False
print(f"SAM charge. VRAM: {vram_info()}")

# Calculer le score de granularité
def get_granularity_score(class_map, sam_masks):
  if len(sam_masks) == 0:
      return float('inf')

  gt_objects = 0
  for cls_id in np.unique(class_map):
      if cls_id < 0: continue
      binary = (class_map == cls_id).astype(np.uint8)
      n_labels, _ = cv2.connectedComponents(binary)
      gt_objects += (n_labels - 1)

  if gt_objects == 0:
      return float('inf')

  mask_purities = []
  for mask_dict in sam_masks:
      m = mask_dict['segmentation'] 
      area = mask_dict['area']

      gt_pixels_under_mask = class_map[m]

      if len(gt_pixels_under_mask) == 0:
          continue

      _, counts = np.unique(gt_pixels_under_mask, return_counts=True)
      majority_class_pixels = counts.max()

      purity = majority_class_pixels / area
      mask_purities.append(purity)

  if not mask_purities:
      return float('inf')

  mean_purity = np.mean(mask_purities)
  under_seg_error = 1.0 - mean_purity

  sam_count = len(sam_masks)
  fragmentation_error = abs(np.log(sam_count / max(gt_objects, 1)))

  final_score = (under_seg_error * 3.0) + (fragmentation_error * 1.0)

  return final_score

import pickle
import os

CACHE_FILE = 'grid_search_cache.pkl'
valeur = 1  

if valeur==1:

    print(f"\n[CACHE] Fichier {CACHE_FILE} trouvé ! Chargement des données...")
    with open(CACHE_FILE, 'rb') as f:
        training_data = pickle.load(f)
    print(f"[CACHE] {len(training_data)} exemples chargés instantanément.")
    
else:
    print(f"\n[CALCUL] Aucun cache trouvé. Démarrage du Grid Search (Ceci va prendre du temps)...")
    PPS_GRID  = [8, 16, 24, 32]
    IOU_GRID  = [0.80, 0.88, 0.95]
    STAB_GRID = [0.82, 0.92]

    N_SAMPLES = len(train_pairs)
    sample_pairs = random.sample(train_pairs, N_SAMPLES)
    n_combos = len(PPS_GRID) * len(IOU_GRID) * len(STAB_GRID)
    print(f"Grid search: {N_SAMPLES} images x {n_combos} combinaisons")

    training_data = []
    for idx, (img_path, lbl_path) in enumerate(tqdm(sample_pairs, desc="Grid search")):
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        class_map = label_to_classmap(lbl_path)
        if class_map is None: continue

        pil_img = Image.fromarray(image_rgb)
        feat_t = resnet_transform(pil_img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            features = feat_extractor(feat_t).cpu().numpy().flatten()

        best_score, best_params = float('inf'), [16.0, 0.86, 0.86]
        for pps in PPS_GRID:
            for iou_t in IOU_GRID:
                for stab_t in STAB_GRID:
                    try:
                        gen = SamAutomaticMaskGenerator(
                            model=sam, points_per_side=pps,
                            pred_iou_thresh=iou_t,
                            stability_score_thresh=stab_t,
                            min_mask_region_area=MIN_AREA)
                        
                        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                            masks = gen.generate(image_rgb)
                            
                        score = get_granularity_score(class_map, masks)
                        if score < best_score:
                            best_score = score
                            best_params = [float(pps), iou_t, stab_t]
                    except: continue

        training_data.append((features, best_params, best_score))
        if (idx + 1) % 5 == 0: cleanup()

    print(f"\n[SAUVEGARDE] Enregistrement des données dans {CACHE_FILE}...")
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(training_data, f)
    
    print(f"Donnees: {len(training_data)} exemples")

valid_scores = [d[2] for d in training_data if d[2] < float('inf')]
if valid_scores:
    print(f"Score moyen optimal: {np.mean(valid_scores):.4f}")
    
X_train_mlp = torch.FloatTensor(np.array([d[0] for d in training_data])).to(DEVICE)
Y_train_mlp = torch.FloatTensor(np.array([d[1] for d in training_data])).to(DEVICE)

MLP_CACHE_FILE = 'mlp_checkpoint.pth'
valeur_2= 1 

if valeur_2==1:
    print(f"\n[CACHE] Fichier {MLP_CACHE_FILE} trouvé ! Chargement du MLP...")
    checkpoint = torch.load(MLP_CACHE_FILE, map_location=DEVICE, weights_only=False)
    mlp.load_state_dict(checkpoint['model_state_dict'])
    losses_mlp = checkpoint['losses']
    mlp.eval()
    print("MLP prêt (chargé depuis le cache).")

else:
    print(f"\n[CALCUL] Aucun cache trouvé. Démarrage de l'entraînement du MLP...")
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)

    losses_mlp = []
    best_loss, best_state_mlp = float('inf'), None
    patience_cnt = 0

    mlp.train()
    for epoch in range(400):
        optimizer.zero_grad()
        pred = mlp(X_train_mlp)
        loss = (F.mse_loss(pred[:, 0], Y_train_mlp[:, 0]) / 784.0 +
                F.mse_loss(pred[:, 1], Y_train_mlp[:, 1]) +
                F.mse_loss(pred[:, 2], Y_train_mlp[:, 2]))
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses_mlp.append(loss.item())

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state_mlp = {k: v.clone() for k, v in mlp.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            
        if patience_cnt >= 40:
            print(f"Early stopping epoch {epoch}")
            break
        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1}: loss={loss.item():.6f}")

    mlp.load_state_dict(best_state_mlp)
    mlp.eval()
    print(f"MLP entraîné. Best loss: {best_loss:.6f}")

    print(f"[SAUVEGARDE] Enregistrement des poids dans {MLP_CACHE_FILE}...")
    torch.save({
        'model_state_dict': mlp.state_dict(),
        'losses': losses_mlp
    }, MLP_CACHE_FILE)

# ============================================================
# EVALUATION DU MLP DE GRANULARITE
# ============================================================
mlp.eval()
with torch.no_grad():
    preds_mlp = mlp(X_train_mlp).cpu().numpy()
Y_np = Y_train_mlp.cpu().numpy()

fig, axes = plt.subplots(1, 4, figsize=(22, 4))
axes[0].plot(losses_mlp, color='#2196F3', alpha=0.8, linewidth=1.5)
axes[0].set_title('Courbe de Loss MLP', fontweight='bold')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
axes[0].set_yscale('log'); axes[0].grid(True, alpha=0.3)

param_names = ['points_per_side', 'pred_iou_thresh', 'stability_score_thresh']
colors_p = ['#E91E63', '#4CAF50', '#FF9800']
for i, (pn, pc) in enumerate(zip(param_names, colors_p)):
    ax = axes[i + 1]
    ax.scatter(Y_np[:, i], preds_mlp[:, i], alpha=0.7, s=40, color=pc,
              edgecolors='white', linewidth=0.5)
    lims = [min(Y_np[:,i].min(), preds_mlp[:,i].min()) - 0.5,
            max(Y_np[:,i].max(), preds_mlp[:,i].max()) + 0.5]
    ax.plot(lims, lims, 'k--', alpha=0.4)
    mae = np.mean(np.abs(Y_np[:, i] - preds_mlp[:, i]))
    ax.set_title(f'{pn}\nMAE={mae:.4f}', fontweight='bold', fontsize=10)
    ax.set_xlabel('Optimal'); ax.set_ylabel('Prediction')
    ax.grid(True, alpha=0.3)

plt.suptitle('Phase 1.2 - Evaluation du MLP de Granularite', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.savefig("fig1.png")

# ============================================================
# CHARGEMENT DINOv2 + PIPELINE PHASE 1
# ============================================================
print("Chargement DINOv2 ViT-S/14...")
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
dinov2.to(DEVICE).eval()
for p in dinov2.parameters(): p.requires_grad = False
DINO_DIM = 384
print(f"DINOv2 charge. Embedding dim: {DINO_DIM} | VRAM: {vram_info()}")

dino_transform = T.Compose([
    T.Resize((224, 224)), T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class Phase1Pipeline:
    # Initialiser le pipeline Phase 1
    def __init__(self, sam_model, mlp_model, feat_ext, dinov2_model):
        self.sam = sam_model
        self.mlp = mlp_model
        self.feat_ext = feat_ext
        self.dinov2 = dinov2_model
        self.classifier = None
        self.classifier_type = None

    # Prédire les paramètres SAM pour l'image
    def predict_sam_params(self, image_rgb):
        pil = Image.fromarray(image_rgb)
        t = resnet_transform(pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            p = self.mlp(self.feat_ext(t)).cpu().numpy()[0]
        return {
            'points_per_side': max(4, int(round(p[0]))),
            'pred_iou_thresh': float(np.clip(p[1], 0.70, 0.98)),
            'stability_score_thresh': float(np.clip(p[2], 0.70, 0.98)),
        }

    # Segmenter l'image
    def segment(self, image_rgb, params=None):
        if params is None:
            params = self.predict_sam_params(image_rgb)
        gen = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=params['points_per_side'],
            pred_iou_thresh=params['pred_iou_thresh'],
            stability_score_thresh=params['stability_score_thresh'],
            min_mask_region_area=MIN_AREA)
        masks = gen.generate(image_rgb)
        return [m for m in masks if m['area'] >= MIN_AREA]

    # Extraire l'embedding d'une bounding box
    def extract_embedding(self, image_rgb, bbox):
        x, y, w, h = [int(v) for v in bbox]
        patch = image_rgb[y:y+h, x:x+w]
        if patch.size == 0 or min(patch.shape[:2]) < 2: return None
        pil = Image.fromarray(patch)
        t = dino_transform(pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            return self.dinov2(t).cpu().numpy()[0]

    # Processus complet
    def process_image(self, image_rgb):
        masks = self.segment(image_rgb)
        results = []
        h, w = image_rgb.shape[:2]
        for m in masks:
            emb = self.extract_embedding(image_rgb, m['bbox'])
            if emb is None: continue
            seg = m['segmentation']
            ys, xs = np.where(seg)
            cx, cy = xs.mean() / w, ys.mean() / h
            cls = self.classify(emb) if self.classifier else -1
            results.append({
                'mask': seg, 'bbox': m['bbox'], 'area': m['area'],
                'embedding': emb, 'centroid': (cx, cy),
                'class': cls,
                'class_name': CLASSES[cls] if 0 <= cls < NUM_CLASSES else 'Unknown',
            })
        return results

    # Classifier un embedding
    def classify(self, embedding):
        if self.classifier is None: return -1
        emb = embedding.reshape(1, -1)
        if self.classifier_type == 'linear':
            return int(self.classifier.predict(emb)[0])
        elif self.classifier_type == 'kmeans':
            cluster = int(self.classifier['model'].predict(emb)[0])
            return self.classifier['mapping'].get(cluster, 0)
        elif self.classifier_type == 'cosine':
            emb_n = embedding / (np.linalg.norm(embedding) + 1e-8)
            sims = np.dot(self.classifier, emb_n)
            return int(np.argmax(sims))
        return -1

pipeline = Phase1Pipeline(sam, mlp, feat_extractor, dinov2)
print("Pipeline Phase 1 initialise.")

# ============================================================
# CONSTRUCTION DES DONNEES + 3 CLASSIFIEURS DINOv2
# ============================================================
print("Extraction des embeddings DINOv2 sur CamVid...")

DINO_CACHE_FILE = 'dino_embeddings_cache.pkl'

valeur_2= 1  
if valeur_2==1:
    print(f"\n[CACHE] Fichier {DINO_CACHE_FILE} trouvé ! Chargement des embeddings DINOv2...")
    with open(DINO_CACHE_FILE, 'rb') as f:
        cache_data = pickle.load(f)
        X_emb = cache_data['X_emb']
        Y_cls = cache_data['Y_cls']
    print(f"[CACHE] {len(X_emb)} embeddings chargés instantanément.")

else:
    print(f"\n[CALCUL] Aucun cache trouvé. Démarrage de l'extraction (Ceci va prendre du temps)...")
    N_CLASSIF = min(20000, len(train_pairs))
    classif_pairs = random.sample(train_pairs, N_CLASSIF)

    all_embeddings, all_labels = [], []
    for img_path, lbl_path in tqdm(classif_pairs, desc="Embeddings"):
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        class_map = label_to_classmap(lbl_path)
        if class_map is None: continue
        masks = pipeline.segment(image_rgb)
        for m in masks:
            seg = m['segmentation']
            mask_cls = class_map[seg]
            valid = mask_cls[mask_cls >= 0]
            if len(valid) == 0: continue
            gt_cls = Counter(valid.tolist()).most_common(1)[0][0]
            emb = pipeline.extract_embedding(image_rgb, m['bbox'])
            if emb is None: continue
            all_embeddings.append(emb)
            all_labels.append(gt_cls)
        cleanup()

    X_emb = np.array(all_embeddings)
    Y_cls = np.array(all_labels)
    
    print(f"\n[SAUVEGARDE] Enregistrement des embeddings dans {DINO_CACHE_FILE}...")
    with open(DINO_CACHE_FILE, 'wb') as f:
        pickle.dump({'X_emb': X_emb, 'Y_cls': Y_cls}, f)

print(f"\nDataset classification: {len(X_emb)} echantillons")
for c in range(NUM_CLASSES):
    cnt = (Y_cls == c).sum()
    if cnt > 0: print(f"  {CLASSES[c]}: {cnt} ({100*cnt/len(Y_cls):.1f}%)")

class_counts = Counter(Y_cls.tolist())
valid_mask = np.array([class_counts[y] >= 3 for y in Y_cls])
X_emb_f, Y_cls_f = X_emb[valid_mask], Y_cls[valid_mask]

X_tr, X_te, Y_tr, Y_te = train_test_split(
    X_emb_f, Y_cls_f, test_size=0.25, random_state=SEED, stratify=Y_cls_f)
print(f"Train: {len(X_tr)}, Test: {len(X_te)}")

print("\n" + "="*60 + "\n METHODE 1: K-Means Clustering\n" + "="*60)
kmeans = MiniBatchKMeans(n_clusters=NUM_CLASSES, random_state=SEED, batch_size=256)
kmeans.fit(X_tr)
km_pred_tr = kmeans.predict(X_tr)
cluster_to_class = {}
for c in range(NUM_CLASSES):
    mask = km_pred_tr == c
    if mask.sum() > 0:
        cluster_to_class[c] = Counter(Y_tr[mask].tolist()).most_common(1)[0][0]
    else: cluster_to_class[c] = c
km_pred_te = np.array([cluster_to_class.get(int(p), 0) for p in kmeans.predict(X_te)])
km_acc = accuracy_score(Y_te, km_pred_te)
km_f1 = f1_score(Y_te, km_pred_te, average='weighted', zero_division=0)
print(f"  Accuracy: {km_acc:.4f} | F1: {km_f1:.4f}")

print("\n" + "="*60 + "\n METHODE 2: Linear Probe\n" + "="*60)
linear_probe = LogisticRegression(max_iter=2000, random_state=SEED,
                                   multi_class='multinomial', C=1.0)
linear_probe.fit(X_tr, Y_tr)
lp_pred_te = linear_probe.predict(X_te)
lp_acc = accuracy_score(Y_te, lp_pred_te)
lp_f1 = f1_score(Y_te, lp_pred_te, average='weighted', zero_division=0)
print(f"  Accuracy: {lp_acc:.4f} | F1: {lp_f1:.4f}")

print("\n" + "="*60 + "\n METHODE 3: Similarite Cosinus\n" + "="*60)
prototypes = np.zeros((NUM_CLASSES, X_tr.shape[1]))
for c in range(NUM_CLASSES):
    mask = Y_tr == c
    if mask.sum() > 0:
        proto = X_tr[mask].mean(axis=0)
        prototypes[c] = proto / (np.linalg.norm(proto) + 1e-8)
cos_preds = []
for emb in X_te:
    emb_n = emb / (np.linalg.norm(emb) + 1e-8)
    cos_preds.append(np.argmax(np.dot(prototypes, emb_n)))
cos_preds = np.array(cos_preds)
cos_acc = accuracy_score(Y_te, cos_preds)
cos_f1 = f1_score(Y_te, cos_preds, average='weighted', zero_division=0)
print(f"  Accuracy: {cos_acc:.4f} | F1: {cos_f1:.4f}")

results_classif = {
    'K-Means': {'acc': km_acc, 'f1': km_f1, 'preds': km_pred_te},
    'Linear Probe': {'acc': lp_acc, 'f1': lp_f1, 'preds': lp_pred_te},
    'Cosine Similarity': {'acc': cos_acc, 'f1': cos_f1, 'preds': cos_preds},
}
print("\n" + "="*60 + "\n COMPARAISON\n" + "="*60)
for name, r in results_classif.items():
    print(f"  {name:20s}: Acc={r['acc']:.4f} | F1={r['f1']:.4f}")
best_method = max(results_classif, key=lambda k: results_classif[k]['f1'])
print(f"\n>>> Meilleure methode: {best_method} (F1={results_classif[best_method]['f1']:.4f})")

if best_method == 'K-Means':
    pipeline.classifier = {'model': kmeans, 'mapping': cluster_to_class}
    pipeline.classifier_type = 'kmeans'
elif best_method == 'Linear Probe':
    pipeline.classifier = linear_probe
    pipeline.classifier_type = 'linear'
else:
    pipeline.classifier = prototypes
    pipeline.classifier_type = 'cosine'

fig, axes = plt.subplots(1, 3, figsize=(20, 5))
for idx, (method, res) in enumerate(results_classif.items()):
    ax = axes[idx]
    present = sorted(set(Y_te.tolist()) | set(res['preds'].tolist()))
    cm = confusion_matrix(Y_te, res['preds'], labels=present)
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title(f"{method}\nAcc={res['acc']:.3f} F1={res['f1']:.3f}", fontweight='bold')
    ax.set_xlabel('Predit'); ax.set_ylabel('Reel')
    tl = [CLASSES[c] if c < NUM_CLASSES else '?' for c in present]
    ax.set_xticks(range(len(present))); ax.set_xticklabels(tl, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(len(present))); ax.set_yticklabels(tl, fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.6)
plt.suptitle('Comparaison des 3 methodes DINOv2', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.savefig("fig2.png")

print(f"\nRapport - {best_method}:")
bp = results_classif[best_method]['preds']
present = sorted(set(Y_te.tolist()) | set(bp.tolist()))
tn = [CLASSES[c] for c in present if c < NUM_CLASSES]
print(classification_report(Y_te, bp, labels=present, target_names=tn, zero_division=0))

# ============================================================
# EVALUATION mIoU — 3 CLASSIFIEURS
# ============================================================
print("\nÉvaluation mIoU pour les 3 classifieurs...")
print("Evaluation mIoU sur CamVid test set...")
N_TEST = min(200, len(test_pairs))
test_subset = random.sample(test_pairs, N_TEST)

# Evaluer le mIoU
def evaluate_miou_closed(classifier, classifier_type, test_subset, label):
    class_ious = defaultdict(list)
    for img_path, lbl_path in tqdm(test_subset, desc=f"mIoU {label}"):
        image     = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gt_map    = label_to_classmap(lbl_path)
        if gt_map is None: continue
        h, w = gt_map.shape
        results  = pipeline.process_image(image_rgb)
        pred_map = np.full((h, w), VOID, dtype=np.int32)
        for r in sorted(results, key=lambda x: x['area'], reverse=True):
            mask = r['mask']
            if mask.shape != (h, w):
                mask = cv2.resize(mask.astype(np.uint8), (w, h),
                                  interpolation=cv2.INTER_NEAREST) > 0
            emb = r['embedding']
            if classifier_type == 'linear':
                cls = int(classifier.predict(emb.reshape(1, -1))[0])
            elif classifier_type == 'kmeans':
                cluster = int(classifier['model'].predict(emb.reshape(1, -1))[0])
                cls = classifier['mapping'].get(cluster, 0)
            elif classifier_type == 'cosine':
                emb_n = emb / (np.linalg.norm(emb) + 1e-8)
                cls   = int(np.argmax(np.dot(classifier, emb_n)))
            pred_map[mask] = cls
        for cls_id in range(NUM_CLASSES):
            gt_bin   = (gt_map   == cls_id)
            pred_bin = (pred_map == cls_id)
            inter    = np.logical_and(gt_bin, pred_bin).sum()
            union    = np.logical_or(gt_bin, pred_bin).sum()
            if union > 0:
                class_ious[cls_id].append(inter / union)
        cleanup()
    ious_dict = {c: np.mean(v) for c, v in class_ious.items() if v}
    miou      = np.mean(list(ious_dict.values())) if ious_dict else 0.0
    return ious_dict, miou

classifiers_eval = {
    'K-Means':           ({'model': kmeans, 'mapping': cluster_to_class}, 'kmeans'),
    'Linear Probe':      (linear_probe,                                    'linear'),
    'Cosine Similarity': (prototypes,                                      'cosine'),
}

all_ious_closed  = {}
all_mious_closed = {}
for name, (clf, ctype) in classifiers_eval.items():
    ious_dict, miou = evaluate_miou_closed(clf, ctype, test_subset, name)
    all_ious_closed[name]  = ious_dict
    all_mious_closed[name] = miou
    print(f"  {name:20s} : mIoU = {miou:.4f}")

fig, axes = plt.subplots(1, 3, figsize=(22, 5), sharey=True)
colors_bar = plt.cm.Set3(np.linspace(0, 1, NUM_CLASSES))

for ax, (method_name, ious_dict) in zip(axes, all_ious_closed.items()):
    miou = all_mious_closed[method_name]
    cls_names = [CLASSES[c] for c in sorted(ious_dict.keys())]
    cls_vals  = [ious_dict[c] for c in sorted(ious_dict.keys())]
    bars = ax.bar(cls_names, cls_vals, color=colors_bar[:len(cls_names)],
                  edgecolor='gray', linewidth=0.5)
    ax.axhline(miou, color='red', linestyle='--', linewidth=2,
               label=f'mIoU = {miou:.4f}')
    ax.set_title(f'{method_name}\nmIoU = {miou:.4f}', fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_ylabel('IoU')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    for b, v in zip(bars, cls_vals):
        ax.text(b.get_x() + b.get_width()/2., b.get_height() + 0.02,
                f'{v:.3f}', ha='center', va='bottom', fontsize=7)
    ax.set_xticklabels(cls_names, rotation=45, ha='right', fontsize=8)

plt.suptitle('Phase 1 — IoU par classe pour les 3 classifieurs (Closed vocabulary)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("fig_miou_3classifiers_closed_k_mean55.png", dpi=150)
plt.close()
print("Graphique sauvegardé : fig_miou_3classifiers_closed.png")

# ============================================================
# EVALUATION mIoU SUR LE TEST SET CamVid
# ============================================================

class_ious = defaultdict(list)
for img_path, lbl_path in tqdm(test_subset, desc="mIoU"):
    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gt_map = label_to_classmap(lbl_path)
    if gt_map is None: continue
    h, w = gt_map.shape
    results = pipeline.process_image(image_rgb)
    pred_map = np.full((h, w), VOID, dtype=np.int32)
    for r in sorted(results, key=lambda x: x['area'], reverse=True):
        mask = r['mask']
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h),
                              interpolation=cv2.INTER_NEAREST) > 0
        pred_map[mask] = r['class']
    for cls_id in range(NUM_CLASSES):
        gt_bin = (gt_map == cls_id)
        pred_bin = (pred_map == cls_id)
        inter = np.logical_and(gt_bin, pred_bin).sum()
        union = np.logical_or(gt_bin, pred_bin).sum()
        if union > 0:
            class_ious[cls_id].append(inter / union)
    cleanup()

print("\n" + "="*60 + "\n RESULTATS mIoU - PHASE 1\n" + "="*60)
ious_dict = {}
for cls_id in range(NUM_CLASSES):
    if cls_id in class_ious and class_ious[cls_id]:
        mean_iou = np.mean(class_ious[cls_id])
        ious_dict[cls_id] = mean_iou
        print(f"  {CLASSES[cls_id]:15s}: IoU = {mean_iou:.4f}")
miou = np.mean(list(ious_dict.values())) if ious_dict else 0
print(f"\n>>> mIoU global: {miou:.4f}")

fig, ax = plt.subplots(figsize=(12, 5))
cn = [CLASSES[c] for c in sorted(ious_dict.keys())]
cv = [ious_dict[c] for c in sorted(ious_dict.keys())]
cb = plt.cm.Set3(np.linspace(0, 1, len(cn)))
bars = ax.bar(cn, cv, color=cb, edgecolor='gray', linewidth=0.5)
ax.axhline(y=miou, color='red', linestyle='--', linewidth=2, label=f'mIoU = {miou:.4f}')
ax.set_ylabel('IoU', fontsize=12); ax.set_ylim(0, 1)
ax.set_title(f'Phase 1 - IoU par classe (mIoU = {miou:.4f})', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
for b, v in zip(bars, cv):
    ax.text(b.get_x() + b.get_width()/2., b.get_height() + 0.02,
            f'{v:.3f}', ha='center', va='bottom', fontsize=8)
plt.xticks(rotation=45, ha='right'); plt.tight_layout(); plt.savefig("fig3.png")

# ============================================================
# VISUALISATION PHASE 1
# ============================================================
CLASS_COLORS = plt.cm.tab20(np.linspace(0, 1, NUM_CLASSES))
n_vis = min(4, N_TEST)
vis_pairs = random.sample(test_subset, n_vis)

fig, axes = plt.subplots(n_vis, 4, figsize=(24, 6 * n_vis))
if n_vis == 1: axes = axes[np.newaxis, :]

for row, (img_path, lbl_path) in enumerate(vis_pairs):
    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gt_map = label_to_classmap(lbl_path)
    results = pipeline.process_image(image_rgb)

    axes[row, 0].imshow(image_rgb)
    axes[row, 0].set_title('Image originale', fontweight='bold'); axes[row, 0].axis('off')

    overlay = image_rgb.copy().astype(np.float32) / 255
    for r in results:
        c = np.random.rand(3)
        overlay[r['mask']] = overlay[r['mask']] * 0.5 + c * 0.5
    axes[row, 1].imshow(overlay)
    axes[row, 1].set_title(f'Masques SAM ({len(results)})', fontweight='bold')
    axes[row, 1].axis('off')

    seg_ov = image_rgb.copy().astype(np.float32) / 255
    for r in results:
        c = r['class']
        if 0 <= c < NUM_CLASSES:
            seg_ov[r['mask']] = seg_ov[r['mask']] * 0.35 + CLASS_COLORS[c][:3] * 0.65
    axes[row, 2].imshow(seg_ov)
    axes[row, 2].set_title('Classes DINOv2', fontweight='bold'); axes[row, 2].axis('off')

    if gt_map is not None:
        gt_vis = np.zeros((*gt_map.shape, 3))
        for cls_id in range(NUM_CLASSES):
            gt_vis[gt_map == cls_id] = CLASS_COLORS[cls_id][:3]
        axes[row, 3].imshow(gt_vis)
    axes[row, 3].set_title('Verite terrain', fontweight='bold'); axes[row, 3].axis('off')
    cleanup()

legend_el = [mpatches.Patch(facecolor=CLASS_COLORS[i][:3], label=CLASSES[i])
             for i in range(NUM_CLASSES)]
fig.legend(handles=legend_el, loc='lower center', ncol=NUM_CLASSES, fontsize=9,
           bbox_to_anchor=(0.5, -0.02))
plt.suptitle('Phase 1 - Segmentation Semantique (MLP + SAM + DINOv2)',
             fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0.04, 1, 0.97]); plt.savefig("fig4.png")

print("\n=== PHASE 1 TERMINEE ===")
print(f"mIoU: {miou:.4f} | Classifieur: {best_method}")

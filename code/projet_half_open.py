# ============================================================
# IMPORTS + CONFIG
# ============================================================
import os, gc, glob, random, warnings, time, csv, subprocess, pickle
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
import torchvision.transforms as T
import torchvision.models as models
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.loader import DataLoader as PyGDataLoader

warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED   = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if DEVICE.type == 'cuda':
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

MIN_AREA = 500
VOID     = -1

def cleanup():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def vram_info():
    if torch.cuda.is_available():
        a = torch.cuda.memory_allocated() / 1e9
        r = torch.cuda.memory_reserved() / 1e9
        return f"Alloc: {a:.2f}GB | Reserve: {r:.2f}GB"
    return "CPU"

print(f"Device: {DEVICE}")

# ============================================================
# CLASSES COCO (80) + CAMVID (11) + MAPPING
# ============================================================
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]
NUM_COCO_CLASSES = len(COCO_CLASSES)  # 80

# IDs COCO officiels (1-based dans les annotations JSON)
# Nécessaire car COCO saute certains IDs (ex: pas de 12, 26, etc.)
COCO_ID_TO_IDX = {
    1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8, 10:9,
    11:10, 13:11, 14:12, 15:13, 16:14, 17:15, 18:16, 19:17, 20:18, 21:19,
    22:20, 23:21, 24:22, 25:23, 27:24, 28:25, 31:26, 32:27, 33:28, 34:29,
    35:30, 36:31, 37:32, 38:33, 39:34, 40:35, 41:36, 42:37, 43:38, 44:39,
    46:40, 47:41, 48:42, 49:43, 50:44, 51:45, 52:46, 53:47, 54:48, 55:49,
    56:50, 57:51, 58:52, 59:53, 60:54, 61:55, 62:56, 63:57, 64:58, 65:59,
    67:60, 70:61, 72:62, 73:63, 74:64, 75:65, 76:66, 77:67, 78:68, 79:69,
    80:70, 81:71, 82:72, 84:73, 85:74, 86:75, 87:76, 88:77, 89:78, 90:79,
}

CAMVID_CLASSES = ['Sky', 'Building', 'Pole', 'Road', 'Sidewalk',
                  'Tree', 'Sign', 'Fence', 'Car', 'Pedestrian', 'Bicyclist']
NUM_CAMVID = len(CAMVID_CLASSES)

# Mapping COCO (index 0-based dans COCO_CLASSES) → CamVid (0-10)
COCO_TO_CAMVID = {
    COCO_CLASSES.index("person"):        9,   # Pedestrian
    COCO_CLASSES.index("bicycle"):      10,   # Bicyclist
    COCO_CLASSES.index("car"):           8,   # Car
    COCO_CLASSES.index("motorcycle"):    8,   # Car
    COCO_CLASSES.index("bus"):           8,   # Car
    COCO_CLASSES.index("train"):         8,   # Car
    COCO_CLASSES.index("truck"):         8,   # Car
    COCO_CLASSES.index("traffic light"): 2,   # Pole
    COCO_CLASSES.index("fire hydrant"):  2,   # Pole
    COCO_CLASSES.index("stop sign"):     6,   # Sign
    COCO_CLASSES.index("parking meter"): 2,   # Pole
    COCO_CLASSES.index("bench"):         4,   # Sidewalk
    COCO_CLASSES.index("bird"):          5,   # Tree (végétation/nature)
    COCO_CLASSES.index("backpack"):      9,   # Pedestrian (porté par personne)
    COCO_CLASSES.index("umbrella"):      9,   # Pedestrian
    COCO_CLASSES.index("handbag"):       9,   # Pedestrian
    COCO_CLASSES.index("suitcase"):      9,   # Pedestrian
    COCO_CLASSES.index("skateboard"):   10,   # Bicyclist
    COCO_CLASSES.index("boat"):          8,   # Car (véhicule)
    COCO_CLASSES.index("airplane"):      8,   # Car (véhicule)
    COCO_CLASSES.index("potted plant"):  5,   # Tree
}

print(f"COCO_TO_CAMVID : {len(COCO_TO_CAMVID)} entrées")
print(f"Configuration : {NUM_COCO_CLASSES} classes COCO | {NUM_CAMVID} classes CamVid évaluation")

# ============================================================
# CHARGEMENT CAMVID
# ============================================================
if not os.path.exists('CamVid'):
    subprocess.run(["git", "clone", "https://github.com/lih627/CamVid.git"])
CAMVID = 'CamVid'

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

COLOR_MAP = {}
csv_path = os.path.join(CAMVID, 'class_dict.csv')
if os.path.exists(csv_path):
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name    = row['name'].strip()
            r, g, b = int(row['r']), int(row['g']), int(row['b'])
            COLOR_MAP[(r, g, b)] = CLASS_GROUPING.get(name, VOID)
    print(f"COLOR_MAP CSV : {len(COLOR_MAP)} couleurs")
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
    print(f"COLOR_MAP fallback : {len(COLOR_MAP)} couleurs")
print(f"COLOR_MAP contient {len(COLOR_MAP)} entrées")

def label_to_classmap(label_path):
    img = cv2.imread(label_path)
    if img is None: return None
    rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    cmap = np.full((h, w), VOID, dtype=np.int32)
    for color, cls in COLOR_MAP.items():
        mask       = np.all(rgb == np.array(color, dtype=np.uint8), axis=-1)
        cmap[mask] = cls
    return cmap

# Split CamVid — rng isolé pour reproductibilité
img_d    = os.path.join(CAMVID, 'CamVid_RGB')
lbl_d    = os.path.join(CAMVID, 'CamVid_Label')
all_pairs = []
if os.path.exists(img_d) and os.path.exists(lbl_d):
    for ip in sorted(glob.glob(os.path.join(img_d, '*.png'))):
        bn = os.path.basename(ip)
        for cand in [os.path.join(lbl_d, bn.replace('.png', '_L.png')),
                     os.path.join(lbl_d, bn)]:
            if os.path.exists(cand):
                all_pairs.append((ip, cand)); break

rng = random.Random(SEED)
rng.shuffle(all_pairs)
n           = len(all_pairs)
train_pairs = all_pairs[:int(0.7  * n)]
val_pairs   = all_pairs[int(0.7  * n):int(0.85 * n)]
test_pairs  = all_pairs[int(0.85 * n):]
print(f"CamVid — Train:{len(train_pairs)} Val:{len(val_pairs)} Test:{len(test_pairs)}")

# ============================================================
# TELECHARGEMENT COCO 2017
# Via pycocotools — pas d'inscription requise
# pip install pycocotools
# ============================================================
import json
import urllib.request
import zipfile

COCO_DIR = "coco2017"
# ============================================================
# TELECHARGEMENT COCO 2017 — VERSION LEGERE
# Stratégie : annotations complètes + images à la demande
# ============================================================
import json, urllib.request, zipfile

COCO_DIR     = "coco2017"
COCO_IMG_DIR = os.path.join(COCO_DIR, "images", "train2017")
MAX_IMAGES   = 4000   

os.makedirs(COCO_IMG_DIR,                          exist_ok=True)
os.makedirs(os.path.join(COCO_DIR, "annotations"), exist_ok=True)

# ── Étape 1 : Annotations uniquement (~241MB) ──
ann_json = os.path.join(COCO_DIR, "annotations", "instances_train2017.json")
if not os.path.exists(ann_json):
    ann_zip = os.path.join(COCO_DIR, "annotations_trainval2017.zip")
    print("Téléchargement annotations COCO (~241MB)...")
    subprocess.run(["wget", "-q", "-O", ann_zip,
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"])
    with zipfile.ZipFile(ann_zip, 'r') as z:
        z.extractall(COCO_DIR)
    os.remove(ann_zip)
    print("Annotations extraites.")
else:
    print("Annotations COCO déjà présentes.")

# ── Étape 2 : Chargement annotations ──
print("Chargement annotations COCO...")
with open(ann_json) as f:
    coco_ann = json.load(f)

coco_id_to_file = {img['id']: img['file_name'] for img in coco_ann['images']}
coco_id_to_url  = {img['id']: img['coco_url']  for img in coco_ann['images']}

coco_anns_by_img = defaultdict(list)
for ann in coco_ann['annotations']:
    if ann.get('segmentation') and ann['area'] > 400:
        coco_anns_by_img[ann['image_id']].append(ann)

# ── Étape 3 : Sélection des images les plus utiles ──
# Priorité aux images avec beaucoup d'objets urbains mappables vers CamVid
URBAN_IDS = {COCO_ID_TO_IDX[c] for c in [
    1,   # person
    2,   # bicycle
    3,   # car
    4,   # motorcycle
    6,   # bus
    7,   # train
    8,   # truck
    10,  # traffic light
    11,  # fire hydrant
    13,  # stop sign
    14,  # parking meter
]}

def urban_score(img_id):
    """Nombre d'annotations urbaines dans l'image."""
    return sum(1 for a in coco_anns_by_img[img_id]
               if a['category_id'] in URBAN_IDS)

# Trier par score urbain décroissant → garder les MAX_IMAGES meilleures
all_valid_ids = [img_id for img_id, anns in coco_anns_by_img.items()
                 if any(a['category_id'] in COCO_ID_TO_IDX for a in anns)]

all_valid_ids.sort(key=urban_score, reverse=True)
selected_ids = all_valid_ids[:MAX_IMAGES]

urban_counts = [urban_score(i) for i in selected_ids]
print(f"Images sélectionnées : {len(selected_ids)} "
      f"(score urbain moyen : {np.mean(urban_counts):.1f} objets/image)")

# ── Étape 4 : Téléchargement à la demande des images sélectionnées ──
def download_image(img_id):
    """Télécharge une image COCO si elle n'existe pas déjà."""
    fname    = coco_id_to_file[img_id]
    out_path = os.path.join(COCO_IMG_DIR, fname)
    if os.path.exists(out_path):
        return True
    url = coco_id_to_url[img_id]
    try:
        urllib.request.urlretrieve(url, out_path)
        return True
    except Exception as e:
        return False

already = len(glob.glob(os.path.join(COCO_IMG_DIR, "*.jpg")))
to_download = [i for i in selected_ids
               if not os.path.exists(
                   os.path.join(COCO_IMG_DIR, coco_id_to_file[i]))]

print(f"Images déjà présentes : {already} | À télécharger : {len(to_download)}")

if to_download:
    print(f"Téléchargement de {len(to_download)} images (~300-500MB selon sélection)...")
    failed = 0
    for img_id in tqdm(to_download, desc="Téléchargement COCO"):
        if not download_image(img_id):
            failed += 1
    print(f"Téléchargement terminé. Échecs : {failed}/{len(to_download)}")

coco_image_ids = selected_ids
print(f"COCO prêt — {len(coco_image_ids)} images avec annotations valides")


# Chargement des annotations COCO
# print("Chargement annotations COCO...")
# with open(os.path.join(COCO_DIR, "annotations", "instances_train2017.json")) as f:
#     coco_ann = json.load(f)

# # Index : image_id → file_name
# coco_id_to_file = {img['id']: img['file_name'] for img in coco_ann['images']}

# # Index : image_id → liste d'annotations
# coco_anns_by_img = defaultdict(list)
# for ann in coco_ann['annotations']:
#     if ann.get('segmentation') and ann['area'] > 400:
#         coco_anns_by_img[ann['image_id']].append(ann)

# # Garder uniquement les images qui ont au moins une annotation utile
# coco_image_ids = [img_id for img_id, anns in coco_anns_by_img.items()
#                   if any(a['category_id'] in COCO_ID_TO_IDX for a in anns)]
# print(f"COCO — {len(coco_image_ids)} images avec annotations valides")

# ============================================================
# VISUALISATION — EXEMPLES ALEATOIRES COCO
# ============================================================
print("\nVisualisation d'exemples COCO...")

# Palette de couleurs pour les 80 classes COCO
np.random.seed(SEED)
COCO_PALETTE = np.random.randint(50, 230, size=(NUM_COCO_CLASSES, 3), dtype=np.uint8)

def draw_coco_segmentation(image_rgb, anns, img_h, img_w):
    """
    Dessine les masques de segmentation COCO sur l'image.
    Retourne l'image annotée.
    """
    overlay = image_rgb.copy().astype(np.float32)
    seg_map = np.zeros((img_h, img_w, 3), dtype=np.float32)
    has_mask = False

    # Trier par aire décroissante (grands masques en premier)
    anns_sorted = sorted(anns, key=lambda a: a['area'], reverse=True)

    for ann in anns_sorted:
        cat_id = ann['category_id']
        if cat_id not in COCO_ID_TO_IDX:
            continue
        cls_idx = COCO_ID_TO_IDX[cat_id]
        color   = COCO_PALETTE[cls_idx].astype(np.float32)

        seg = ann['segmentation']
        mask = np.zeros((img_h, img_w), dtype=np.uint8)

        if isinstance(seg, list):
            # Polygones
            for poly in seg:
                pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(mask, [pts], 1)
        elif isinstance(seg, dict):
            # RLE (run-length encoding)
            try:
                from pycocotools import mask as maskUtils
                mask = maskUtils.decode(seg)
            except ImportError:
                continue

        if mask.sum() == 0:
            continue

        seg_map[mask > 0] = color
        has_mask = True

    if not has_mask:
        return image_rgb, np.zeros_like(image_rgb)

    # Blend : image originale + masques semi-transparents
    alpha   = 0.5
    blended = (overlay * (1 - alpha) + seg_map * alpha).clip(0, 255).astype(np.uint8)
    seg_only = seg_map.astype(np.uint8)
    return blended, seg_only


def add_legend(ax, anns):
    """Ajoute une légende avec les classes présentes sur l'image."""
    seen = {}
    for ann in anns:
        cat_id = ann['category_id']
        if cat_id not in COCO_ID_TO_IDX or cat_id in seen:
            continue
        cls_idx = COCO_ID_TO_IDX[cat_id]
        seen[cat_id] = (COCO_CLASSES[cls_idx], COCO_PALETTE[cls_idx])

    patches = [
        mpatches.Patch(
            color=np.array(color) / 255.0,
            label=f"{name}" + (" → " + CAMVID_CLASSES[COCO_TO_CAMVID[COCO_CLASSES.index(name)]]
                               if COCO_CLASSES.index(name) in COCO_TO_CAMVID else "")
        )
        for cat_id, (name, color) in seen.items()
    ]
    if patches:
        ax.legend(handles=patches, loc='upper right', fontsize=6,
                  framealpha=0.8, ncol=max(1, len(patches) // 6))


# Sélection d'images aléatoires avec au moins 2 annotations mappables vers CamVid
rng_visu = random.Random(SEED + 50)
N_VISU   = 3   # ← nombre d'images à afficher

urban_ids = [
    img_id for img_id in coco_image_ids
    if sum(1 for a in coco_anns_by_img[img_id]
           if COCO_ID_TO_IDX.get(a['category_id'], -1) in COCO_TO_CAMVID) >= 2
]
sample_visu = rng_visu.sample(urban_ids, min(N_VISU, len(urban_ids)))
print(f"  {len(urban_ids)} images avec ≥2 objets urbains — affichage de {len(sample_visu)}")

fig, axes = plt.subplots(N_VISU, 3, figsize=(18, N_VISU * 4))
fig.suptitle('Exemples COCO — RGB / Segmentation mixte / Segmentation seule',
             fontsize=14, fontweight='bold', y=1.01)

for row, img_id in enumerate(sample_visu):
    fname     = coco_id_to_file[img_id]
    img_path  = os.path.join(COCO_IMG_DIR, fname)
    image     = cv2.imread(img_path)
    # --- AJOUTE CES DEUX LIGNES ---
    if image is None:
        print(f"Attention: l'image {fname} est introuvable. On passe à la suivante.")
        continue
    # ------------------------------
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h_img, w_img = image_rgb.shape[:2]
    anns      = coco_anns_by_img[img_id]

    blended, seg_only = draw_coco_segmentation(image_rgb, anns, h_img, w_img)

    # Colonne 0 — Image RGB originale
    axes[row, 0].imshow(image_rgb)
    axes[row, 0].set_title(f'RGB — {fname}', fontsize=8)
    axes[row, 0].axis('off')

    # Colonne 1 — Image + masques superposés
    axes[row, 1].imshow(blended)
    axes[row, 1].set_title('Segmentation (blend α=0.5)', fontsize=8)
    axes[row, 1].axis('off')
    add_legend(axes[row, 1], anns)

    # Colonne 2 — Masques seuls + classes mappées CamVid en surimpression
    axes[row, 2].imshow(seg_only)
    axes[row, 2].set_title('Masques + mapping CamVid', fontsize=8)
    axes[row, 2].axis('off')

    # Boîtes englobantes avec nom de classe CamVid mappée
    for ann in anns:
        cat_id = ann['category_id']
        if cat_id not in COCO_ID_TO_IDX: continue
        cls_idx = COCO_ID_TO_IDX[cat_id]
        if cls_idx not in COCO_TO_CAMVID: continue
        cam_name = CAMVID_CLASSES[COCO_TO_CAMVID[cls_idx]]
        x, y, bw, bh = ann['bbox']
        rect = plt.Rectangle((x, y), bw, bh,
                              linewidth=1.5, edgecolor='white',
                              facecolor='none', linestyle='--')
        axes[row, 2].add_patch(rect)
        axes[row, 2].text(x + 2, y - 4, cam_name,
                          color='white', fontsize=6, fontweight='bold',
                          bbox=dict(boxstyle='round,pad=0.1',
                                    facecolor='black', alpha=0.6))

plt.tight_layout()
plt.savefig("fig_coco_exemples.png", dpi=120, bbox_inches='tight')
plt.close()
print("Graphique sauvegardé : fig_coco_exemples.png")

COCO_IMG_DIR = os.path.join(COCO_DIR, "images", "train2017")

# SAM
SAM_CKPT = 'sam_vit_h_4b8939.pth'
if not os.path.exists(SAM_CKPT):
    subprocess.run(["wget", "-q",
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"])

# ============================================================
# MLP GRANULARITE + SAM
# ============================================================
class ImageFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        for p in self.parameters(): p.requires_grad = False
        self.eval()
    def forward(self, x):
        with torch.no_grad():
            return self.backbone(x).squeeze(-1).squeeze(-1)

class GranularityMLP(nn.Module):
    def __init__(self, in_dim=512, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden // 2, 3),
        )
    def forward(self, features):
        raw  = self.net(features)
        pps  = torch.sigmoid(raw[:, 0]) * 28.0 + 4.0
        iou  = torch.sigmoid(raw[:, 1]) * 0.28 + 0.70
        stab = torch.sigmoid(raw[:, 2]) * 0.28 + 0.70
        return torch.stack([pps, iou, stab], dim=-1)

feat_extractor   = ImageFeatureExtractor().to(DEVICE)
mlp              = GranularityMLP().to(DEVICE)
resnet_transform = T.Compose([
    T.Resize((224, 224)), T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

sam = sam_model_registry["vit_h"](checkpoint=SAM_CKPT)
sam.to(DEVICE).eval()
for p in sam.parameters(): p.requires_grad = False
print(f"SAM chargé. {vram_info()}")

def get_granularity_score(class_map, sam_masks):
    if len(sam_masks) == 0: return float('inf')
    gt_objects = 0
    for cls_id in np.unique(class_map):
        if cls_id < 0: continue
        binary = (class_map == cls_id).astype(np.uint8)
        n_labels, _ = cv2.connectedComponents(binary)
        gt_objects += (n_labels - 1)
    if gt_objects == 0: return float('inf')
    mask_purities = []
    for md in sam_masks:
        m, area = md['segmentation'], md['area']
        gt_px = class_map[m]
        if len(gt_px) == 0: continue
        _, counts = np.unique(gt_px, return_counts=True)
        mask_purities.append(counts.max() / area)
    if not mask_purities: return float('inf')
    return (1.0 - np.mean(mask_purities)) * 3.0 + abs(np.log(len(sam_masks) / max(gt_objects, 1)))

with open('grid_search_cache.pkl', 'rb') as f: training_data = pickle.load(f)
print(f"[CACHE] {len(training_data)} exemples grid search.")

X_train_mlp = torch.FloatTensor(np.array([d[0] for d in training_data])).to(DEVICE)
Y_train_mlp = torch.FloatTensor(np.array([d[1] for d in training_data])).to(DEVICE)

checkpoint = torch.load('mlp_checkpoint.pth', map_location=DEVICE, weights_only=False)
mlp.load_state_dict(checkpoint['model_state_dict'])
mlp.eval()
print("MLP granularité chargé.")

# ============================================================
# DINOV2
# ============================================================
print("Chargement DINOv2 ViT-S/14...")
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
dinov2.to(DEVICE).eval()
for p in dinov2.parameters(): p.requires_grad = False
DINO_DIM = 384
dino_transform = T.Compose([
    T.Resize((224, 224)), T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
print(f"DINOv2 chargé. {vram_info()}")

# ============================================================
# EXTRACTION EMBEDDINGS COCO
# Stratégie : bounding box de chaque annotation → patch → DINOv2
# ============================================================
COCO_EMB_CACHE = "coco_dino_embeddings.pkl"
RECALC_COCO    = 0

if RECALC_COCO == 0 and os.path.exists(COCO_EMB_CACHE):
    with open(COCO_EMB_CACHE, 'rb') as f: coco_cache = pickle.load(f)
    X_coco, Y_coco = coco_cache['X_emb'], coco_cache['Y_cls']
    print(f"[CACHE] {len(X_coco)} embeddings COCO — {len(np.unique(Y_coco))} classes.")
else:
    print("[CALCUL] Extraction embeddings DINOv2 sur COCO...")
    print(f"  {len(coco_image_ids)} images à traiter...")
    all_embs, all_labels = [], []

    # Limiter à 50 000 images pour la RAM/temps
    sample_ids = coco_image_ids[:50000]

    for img_id in tqdm(sample_ids, desc="COCO→DINOv2"):
        try:
            fname     = coco_id_to_file[img_id]
            img_path  = os.path.join(COCO_IMG_DIR, fname)
            image     = cv2.imread(img_path)
            if image is None: continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h_img, w_img = image_rgb.shape[:2]

            anns = coco_anns_by_img[img_id]
            for ann in anns:
                cat_id = ann['category_id']
                if cat_id not in COCO_ID_TO_IDX: continue
                cls_idx = COCO_ID_TO_IDX[cat_id]

                # Bounding box COCO : [x, y, width, height] en pixels absolus
                x, y, bw, bh = [int(v) for v in ann['bbox']]
                margin = 8
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(w_img, x + bw + margin)
                y2 = min(h_img, y + bh + margin)

                if (x2 - x1) < 8 or (y2 - y1) < 8: continue
                if ann['area'] < 400: continue

                patch = image_rgb[y1:y2, x1:x2]
                if patch.size == 0: continue

                t = dino_transform(Image.fromarray(patch)).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    emb = dinov2(t).cpu().numpy()[0]

                all_embs.append(emb)
                all_labels.append(cls_idx)

        except Exception: continue

        if len(all_embs) % 10000 == 0 and len(all_embs) > 0: cleanup()

    X_coco = np.array(all_embs,   dtype=np.float32)
    Y_coco = np.array(all_labels, dtype=np.int64)
    with open(COCO_EMB_CACHE, 'wb') as f:
        pickle.dump({'X_emb': X_coco, 'Y_cls': Y_coco}, f)
    print(f"  {len(X_coco)} embeddings | {len(np.unique(Y_coco))} classes | sauvegardé.")

# Distribution des classes
print("\nDistribution COCO (top 15 classes) :")
for cls_idx, cnt in Counter(Y_coco.tolist()).most_common(15):
    print(f"  [{cls_idx:2d}] {COCO_CLASSES[cls_idx]:20s} : {cnt:6d} exemples")

# Split train/test COCO
class_counts_coco = Counter(Y_coco.tolist())
valid_mask_coco   = np.array([class_counts_coco[y] >= 3 for y in Y_coco])
X_coco_f, Y_coco_f = X_coco[valid_mask_coco], Y_coco[valid_mask_coco]
X_tr_coco, X_te_coco, Y_tr_coco, Y_te_coco = train_test_split(
    X_coco_f, Y_coco_f, test_size=0.15, random_state=SEED, stratify=Y_coco_f)
print(f"\nCOCO split — Train:{len(X_tr_coco)} Test:{len(X_te_coco)}")

# ============================================================
# 3 CLASSIFIEURS ENTRAINES SUR COCO (80 classes)
# ============================================================

# ── Méthode 1 : K-Means ──
print("\n" + "="*60)
print(" METHODE 1 : K-Means (80 clusters, COCO)")
print("="*60)

KMEANS_CACHE  = "coco_kmeans.pkl"
RECALC_KMEANS = 0

if RECALC_KMEANS == 0 and os.path.exists(KMEANS_CACHE):
    with open(KMEANS_CACHE, 'rb') as f: km_data = pickle.load(f)
    kmeans_coco    = km_data['model']
    cluster_to_coco = km_data['cluster_to_coco']
    print("[CACHE] K-Means COCO chargé.")
else:
    kmeans_coco = MiniBatchKMeans(n_clusters=NUM_COCO_CLASSES,
                                   random_state=SEED, batch_size=1024)
    kmeans_coco.fit(X_tr_coco)
    km_pred_tr = kmeans_coco.predict(X_tr_coco)
    cluster_to_coco = {}
    for c in range(NUM_COCO_CLASSES):
        mask = km_pred_tr == c
        if mask.sum() > 0:
            cluster_to_coco[c] = Counter(Y_tr_coco[mask].tolist()).most_common(1)[0][0]
        else:
            cluster_to_coco[c] = c
    with open(KMEANS_CACHE, 'wb') as f:
        pickle.dump({'model': kmeans_coco, 'cluster_to_coco': cluster_to_coco}, f)
    print("[TRAIN] K-Means COCO entraîné et sauvegardé.")

km_pred_te_coco = np.array([cluster_to_coco.get(int(p), 0)
                              for p in kmeans_coco.predict(X_te_coco)])
km_acc = accuracy_score(Y_te_coco, km_pred_te_coco)
km_f1  = f1_score(Y_te_coco, km_pred_te_coco, average='weighted', zero_division=0)
print(f"  Accuracy COCO : {km_acc:.4f} | F1 : {km_f1:.4f}")

# ── Méthode 2 : Linear Probe (384 → 80) ──
print("\n" + "="*60)
print(" METHODE 2 : Linear Probe PyTorch (384→80, COCO)")
print("="*60)

class LinearProbe(nn.Module):
    def __init__(self, in_dim=DINO_DIM, n_classes=NUM_COCO_CLASSES):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)
    def forward(self, x): return self.fc(x)

probe_model     = LinearProbe().to(DEVICE)
COCO_PROBE_CACHE = "coco_linear_probe.pth"
RECALC_PROBE     = 0

if RECALC_PROBE == 0 and os.path.exists(COCO_PROBE_CACHE):
    ckpt = torch.load(COCO_PROBE_CACHE, map_location=DEVICE)
    probe_model.load_state_dict(ckpt['model_state_dict'])
    probe_model.eval()
    print(f"[CACHE] Linear Probe COCO chargé ({NUM_COCO_CLASSES} classes).")
else:
    print("[TRAIN] Entraînement Linear Probe COCO...")
    MAX_PER_CLS = 3000
    idx_bal     = []
    for c in np.unique(Y_coco):
        idx_c = np.where(Y_coco == c)[0]
        if len(idx_c) > MAX_PER_CLS:
            idx_c = np.random.choice(idx_c, MAX_PER_CLS, replace=False)
        idx_bal.extend(idx_c.tolist())
    random.shuffle(idx_bal)
    X_bal, Y_bal = X_coco[idx_bal], Y_coco[idx_bal]
    counts_bal   = Counter(Y_bal.tolist())
    vm           = np.array([counts_bal[y] >= 5 for y in Y_bal])
    X_bal, Y_bal = X_bal[vm], Y_bal[vm]
    X_tr_p, X_va_p, Y_tr_p, Y_va_p = train_test_split(
        X_bal, Y_bal, test_size=0.15, random_state=SEED, stratify=Y_bal)
    print(f"  Train:{len(X_tr_p)} | Val:{len(X_va_p)}")

    class EmbDataset(Dataset):
        def __init__(self, X, Y):
            self.X = torch.FloatTensor(X); self.Y = torch.LongTensor(Y)
        def __len__(self): return len(self.X)
        def __getitem__(self, i): return self.X[i], self.Y[i]

    tr_ld = DataLoader(EmbDataset(X_tr_p, Y_tr_p), batch_size=512, shuffle=True,  num_workers=2)
    va_ld = DataLoader(EmbDataset(X_va_p, Y_va_p), batch_size=512, shuffle=False, num_workers=2)
    opt_p  = torch.optim.AdamW(probe_model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch_p  = torch.optim.lr_scheduler.CosineAnnealingLR(opt_p, T_max=60)
    crit_p = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_acc_p, best_state_p, patience_p, probe_losses = 0.0, None, 0, []

    for epoch in range(300):
        probe_model.train()
        t_losses = []
        for xb, yb in tr_ld:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt_p.zero_grad()
            loss = crit_p(probe_model(xb), yb)
            loss.backward(); opt_p.step()
            t_losses.append(loss.item())
        sch_p.step()
        probe_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in va_ld:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                correct += (probe_model(xb).argmax(1) == yb).sum().item()
                total   += len(yb)
        v_acc = correct / total
        probe_losses.append(np.mean(t_losses))
        if v_acc > best_acc_p:
            best_acc_p   = v_acc
            best_state_p = {k: v.clone() for k, v in probe_model.state_dict().items()}
            patience_p   = 0
        else:
            patience_p += 1
        if patience_p >= 15:
            print(f"  Early stopping epoch {epoch}"); break
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d} | loss={probe_losses[-1]:.4f} | val_acc={v_acc:.4f}")
    probe_model.load_state_dict(best_state_p)
    probe_model.eval()
    torch.save({'model_state_dict': probe_model.state_dict(),
                'losses': probe_losses}, COCO_PROBE_CACHE)
    print(f"  Best val_acc : {best_acc_p:.4f} | Sauvegardé.")

lp_preds_te = []
probe_model.eval()
with torch.no_grad():
    for i in range(0, len(X_te_coco), 512):
        xb = torch.FloatTensor(X_te_coco[i:i+512]).to(DEVICE)
        lp_preds_te.extend(probe_model(xb).argmax(1).cpu().numpy().tolist())
lp_preds_te = np.array(lp_preds_te)
lp_acc = accuracy_score(Y_te_coco, lp_preds_te)
lp_f1  = f1_score(Y_te_coco, lp_preds_te, average='weighted', zero_division=0)
print(f"  Accuracy COCO : {lp_acc:.4f} | F1 : {lp_f1:.4f}")

# ── Méthode 3 : Similarité Cosinus ──
print("\n" + "="*60)
print(" METHODE 3 : Similarité Cosinus (prototypes COCO)")
print("="*60)

PROTO_CACHE  = "coco_prototypes.pkl"
RECALC_PROTO = 0

if RECALC_PROTO == 0 and os.path.exists(PROTO_CACHE):
    with open(PROTO_CACHE, 'rb') as f: prototypes_coco = pickle.load(f)
    print("[CACHE] Prototypes COCO chargés.")
else:
    prototypes_coco = np.zeros((NUM_COCO_CLASSES, DINO_DIM), dtype=np.float32)
    for c in range(NUM_COCO_CLASSES):
        mask = Y_tr_coco == c
        if mask.sum() > 0:
            proto = X_tr_coco[mask].mean(axis=0)
            prototypes_coco[c] = proto / (np.linalg.norm(proto) + 1e-8)
    with open(PROTO_CACHE, 'wb') as f: pickle.dump(prototypes_coco, f)
    print("[TRAIN] Prototypes COCO calculés et sauvegardés.")

cos_preds_te = []
for emb in tqdm(X_te_coco, desc="Cosine eval"):
    emb_n = emb / (np.linalg.norm(emb) + 1e-8)
    cos_preds_te.append(int(np.argmax(np.dot(prototypes_coco, emb_n))))
cos_preds_te = np.array(cos_preds_te)
cos_acc = accuracy_score(Y_te_coco, cos_preds_te)
cos_f1  = f1_score(Y_te_coco, cos_preds_te, average='weighted', zero_division=0)
print(f"  Accuracy COCO : {cos_acc:.4f} | F1 : {cos_f1:.4f}")

# ── Résumé ──
print("\n" + "="*60)
print(" COMPARAISON CLASSIFIEURS — entraînés sur COCO 80 classes")
print("="*60)
for name, acc, f1 in [('K-Means',           km_acc,  km_f1),
                       ('Linear Probe',      lp_acc,  lp_f1),
                       ('Cosine Similarity', cos_acc, cos_f1)]:
    print(f"  {name:20s} : Acc={acc:.4f} | F1={f1:.4f}")

# ============================================================
# PIPELINE HALF-OPEN COCO
# ============================================================
class Phase1PipelineHalfOpen:
    def __init__(self, sam_model, mlp_model, feat_ext, dinov2_model,
                 classifier, classifier_type):
        self.sam             = sam_model
        self.mlp             = mlp_model
        self.feat_ext        = feat_ext
        self.dinov2          = dinov2_model
        self.classifier      = classifier
        self.classifier_type = classifier_type

    def predict_sam_params(self, image_rgb):
        pil = Image.fromarray(image_rgb)
        t   = resnet_transform(pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            p = self.mlp(self.feat_ext(t)).cpu().numpy()[0]
        return {'points_per_side':        max(4, int(round(p[0]))),
                'pred_iou_thresh':        float(np.clip(p[1], 0.70, 0.98)),
                'stability_score_thresh': float(np.clip(p[2], 0.70, 0.98))}

    def segment(self, image_rgb, params=None):
        if params is None: params = self.predict_sam_params(image_rgb)
        gen = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=params['points_per_side'],
            pred_iou_thresh=params['pred_iou_thresh'],
            stability_score_thresh=params['stability_score_thresh'],
            min_mask_region_area=MIN_AREA)
        return [m for m in gen.generate(image_rgb) if m['area'] >= MIN_AREA]

    def extract_embedding(self, image_rgb, bbox):
        x, y, w, h = [int(v) for v in bbox]
        patch = image_rgb[y:y+h, x:x+w]
        if patch.size == 0 or min(patch.shape[:2]) < 2: return None
        t = dino_transform(Image.fromarray(patch)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            return self.dinov2(t).cpu().numpy()[0]

    def classify(self, embedding):
        """Retourne (coco_idx, confidence) — index COCO 0-based."""
        if self.classifier_type == 'linear':
            t = torch.FloatTensor(embedding).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                probs = torch.softmax(self.classifier(t), dim=1).cpu().numpy()[0]
            return int(np.argmax(probs)), float(probs.max())

        elif self.classifier_type == 'kmeans':
            cluster  = int(self.classifier['model'].predict(embedding.reshape(1, -1))[0])
            coco_cls = self.classifier['cluster_to_coco'].get(cluster, 0)
            return coco_cls, 1.0

        elif self.classifier_type == 'cosine':
            emb_n = embedding / (np.linalg.norm(embedding) + 1e-8)
            sims  = np.dot(self.classifier, emb_n)
            idx   = int(np.argmax(sims))
            return idx, float(sims[idx])

        return 0, 0.0

    def process_image(self, image_rgb):
        masks        = self.segment(image_rgb)
        h_img, w_img = image_rgb.shape[:2]
        results      = []
        for m in masks:
            emb = self.extract_embedding(image_rgb, m['bbox'])
            if emb is None: continue
            coco_idx, confidence = self.classify(emb)
            seg    = m['segmentation']
            ys, xs = np.where(seg)
            cx, cy = xs.mean() / w_img, ys.mean() / h_img
            results.append({
                'mask':       seg,
                'bbox':       m['bbox'],
                'area':       m['area'],
                'embedding':  emb,
                'centroid':   (cx, cy),
                'class':      coco_idx,
                'class_name': COCO_CLASSES[coco_idx],
                'confidence': confidence,
            })
        return results

# ============================================================
# EVALUATION mIoU — 3 CLASSIFIEURS × MAPPING COCO→CAMVID
# ============================================================
CLOSED_MIOU = {
    'Sky': 0.769, 'Building': 0.566, 'Pole': 0.116,
    'Road': 0.803, 'Sidewalk': 0.374, 'Tree': 0.495,
    'Sign': 0.306, 'Fence': 0.225,    'Car': 0.488,
    'Pedestrian': 0.235, 'Bicyclist': 0.109,
}

rng_test    = random.Random(SEED)
N_TEST      = min(200, len(test_pairs))
test_subset = rng_test.sample(test_pairs, N_TEST)
print(f"\nTest subset : {N_TEST} images CamVid (identique pour les 3 classifieurs)")

def evaluate_miou(pipeline, test_subset, method_name):
    print(f"\nÉvaluation mIoU — {method_name}...")
    class_ious = defaultdict(list)

    for img_path, lbl_path in tqdm(test_subset, desc=method_name):
        image     = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gt_map    = label_to_classmap(lbl_path)
        if gt_map is None: continue
        h, w = gt_map.shape

        results  = pipeline.process_image(image_rgb)
        pred_map = np.full((h, w), VOID, dtype=np.int32)

        for r in sorted(results, key=lambda x: x['area'], reverse=True):
            coco_cls = r['class']
            cam_cls  = COCO_TO_CAMVID.get(coco_cls, VOID)
            if cam_cls == VOID: continue
            mask = r['mask']
            if mask.shape != (h, w):
                mask = cv2.resize(mask.astype(np.uint8), (w, h),
                                  interpolation=cv2.INTER_NEAREST) > 0
            pred_map[mask] = cam_cls

        for cls_id in range(NUM_CAMVID):
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

pipelines = {
    'K-Means': Phase1PipelineHalfOpen(
        sam, mlp, feat_extractor, dinov2,
        {'model': kmeans_coco, 'cluster_to_coco': cluster_to_coco}, 'kmeans'),
    'Linear Probe': Phase1PipelineHalfOpen(
        sam, mlp, feat_extractor, dinov2,
        probe_model, 'linear'),
    'Cosine Similarity': Phase1PipelineHalfOpen(
        sam, mlp, feat_extractor, dinov2,
        prototypes_coco, 'cosine'),
}

all_ious  = {}
all_mious = {}
for method_name, pipeline in pipelines.items():
    ious_dict, miou = evaluate_miou(pipeline, test_subset, method_name)
    all_ious[method_name]  = ious_dict
    all_mious[method_name] = miou

# ============================================================
# GRAPHIQUE MATRICES DE CONFUSION (classes CamVid mappées)
# ============================================================
print("\nCalcul matrices de confusion...")

def coco_to_cam_array(preds):
    return np.array([COCO_TO_CAMVID.get(int(p), VOID) for p in preds])

mappable_mask = np.array([int(y) in COCO_TO_CAMVID for y in Y_te_coco])
Y_te_cam      = coco_to_cam_array(Y_te_coco[mappable_mask])

km_preds_cam  = coco_to_cam_array(km_pred_te_coco[mappable_mask])
lp_preds_cam  = coco_to_cam_array(lp_preds_te[mappable_mask])
cos_preds_cam = coco_to_cam_array(cos_preds_te[mappable_mask])

def filter_void(y_true, y_pred):
    mask = (y_true != VOID) & (y_pred != VOID)
    return y_true[mask], y_pred[mask]

results_conf = {
    'K-Means':           filter_void(Y_te_cam, km_preds_cam),
    'Linear Probe':      filter_void(Y_te_cam, lp_preds_cam),
    'Cosine Similarity': filter_void(Y_te_cam, cos_preds_cam),
}

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
for ax, (method, (yt, yp)) in zip(axes, results_conf.items()):
    if len(yt) == 0:
        ax.set_title(f'{method}\n(aucune donnée)'); continue
    acc = accuracy_score(yt, yp)
    f1  = f1_score(yt, yp, average='weighted', zero_division=0)
    present = sorted(set(yt.tolist()) | set(yp.tolist()))
    cm  = confusion_matrix(yt, yp, labels=present)
    im  = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title(f'{method}\nAcc={acc:.3f} F1={f1:.3f}', fontweight='bold')
    ax.set_xlabel('Prédit'); ax.set_ylabel('Réel')
    tl = [CAMVID_CLASSES[c] for c in present]
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels(tl, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(len(present)))
    ax.set_yticklabels(tl, fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.6)

plt.suptitle('Comparaison des 3 méthodes DINOv2 — Half-open COCO (→CamVid)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("fig_confusion_halfopen_coco.png", dpi=150)
plt.close()
print("Graphique sauvegardé : fig_confusion_halfopen_coco.png")

# ============================================================
# AFFICHAGE RESULTATS TABLEAU
# ============================================================
print("\n" + "="*75)
print(" RESULTATS mIoU — HALF-OPEN COCO (80 classes) vs CLOSED (CamVid 11)")
print("="*75)

header = f"  {'Classe':15s}  {'Closed':>8s}"
for name in pipelines:
    short = {'K-Means': 'K-Means', 'Linear Probe': 'LinProbe',
             'Cosine Similarity': 'Cosine'}[name]
    header += f"  {short:>9s}  {'Δ':>7s}"
print(header)
print("-" * 75)

for cls_id in range(NUM_CAMVID):
    cls_name = CAMVID_CLASSES[cls_id]
    closed   = CLOSED_MIOU.get(cls_name, 0.0)
    line     = f"  {cls_name:15s}  {closed:8.4f}"
    for name in pipelines:
        iou   = all_ious[name].get(cls_id, 0.0)
        delta = iou - closed
        arrow = "▲" if delta >= 0 else "▼"
        line += f"  {iou:9.4f}  {arrow}{abs(delta):.4f}"
    print(line)

print("-" * 75)
miou_closed = np.mean(list(CLOSED_MIOU.values()))
line_miou   = f"  {'mIoU':15s}  {miou_closed:8.4f}"
for name in pipelines:
    delta = all_mious[name] - miou_closed
    arrow = "▲" if delta >= 0 else "▼"
    line_miou += f"  {all_mious[name]:9.4f}  {arrow}{abs(delta):.4f}"
print(line_miou)

# ============================================================
# GRAPHIQUE COMPARATIF — 4 barres par classe
# ============================================================
fig, ax = plt.subplots(figsize=(16, 6))
x      = np.arange(NUM_CAMVID)
w_bar  = 0.18
colors = ['#90CAF9', '#A5D6A7', '#FFCC80', '#CE93D8']
labels = ['Closed (CamVid 11)', 'Half-open K-Means',
          'Half-open Linear Probe', 'Half-open Cosine']

all_series = [
    [CLOSED_MIOU.get(CAMVID_CLASSES[c], 0.0)      for c in range(NUM_CAMVID)],
    [all_ious['K-Means'].get(c, 0.0)               for c in range(NUM_CAMVID)],
    [all_ious['Linear Probe'].get(c, 0.0)          for c in range(NUM_CAMVID)],
    [all_ious['Cosine Similarity'].get(c, 0.0)     for c in range(NUM_CAMVID)],
]
mious_all   = [miou_closed, all_mious['K-Means'],
               all_mious['Linear Probe'], all_mious['Cosine Similarity']]
line_styles = ['--', '-.', ':', '-']
line_colors = ['#1976D2', '#388E3C', '#E65100', '#6A1B9A']

for i, (serie, color, label, miou_val, ls, lc) in enumerate(
        zip(all_series, colors, labels, mious_all, line_styles, line_colors)):
    offset = (i - 2 + 0.5) * w_bar
    ax.bar(x + offset, serie, w_bar, color=color, edgecolor='gray',
           linewidth=0.5, label=label)
    ax.axhline(miou_val, color=lc, linestyle=ls, linewidth=1.5,
               label=f'mIoU {label.split()[0]}={miou_val:.3f}')

ax.set_xticks(x)
ax.set_xticklabels(CAMVID_CLASSES, rotation=45, ha='right')
ax.set_ylabel('IoU'); ax.set_ylim(0, 1)
ax.set_title('Closed vs Half-open COCO — mIoU par classe et par classifieur',
             fontweight='bold')
ax.legend(loc='upper right', fontsize=7)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig("fig_closed_vs_halfopen_coco.png", dpi=150)
plt.close()
print("Graphique sauvegardé : fig_closed_vs_halfopen_coco.png")

print("\n=== PHASE 1 HALF-OPEN COCO TERMINÉE ===")
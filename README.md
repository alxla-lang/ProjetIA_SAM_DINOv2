# ProjetIA_SAM_DINOv2

Pipeline de segmentation sémantique combinant SAM et DINOv2,
évalué en **vocabulaire fermé** (CamVid, 11 classes) et
**vocabulaire semi-ouvert** (COCO, 80 classes).

## Résultats

| Mode | K-Means | Linear Probe | Cosine Sim. |
|------|---------|--------------|-------------|
| Fermé — CamVid 11 classes | 0.066 | **0.478** | 0.245 |
| Semi-ouvert — COCO 80 classes | 0.010 | 0.025 | 0.036 |

## Structure du dépôt
```
code/
├── projet_IA_closed.py          # Pipeline vocabulaire fermé
├── projet_half_open.py   # Pipeline vocabulaire semi-ouvert (COCO)
└── download_data.py      # Téléchargement automatique des données
scripts/
├── run_closed.sh         # Script SLURM — vocabulaire fermé
└── run_halfopen.sh       # Script SLURM — vocabulaire semi-ouvert
results/                  # Figures et graphiques de résultats
cache/                    # Instructions pour recréer les caches
```

## Installation
```bash
conda create -n projet_ia python=3.10 -y
conda activate projet_ia
pip install -r requirements.txt
```

## Téléchargement des données
```bash
# Tout télécharger (CamVid + SAM + annotations COCO)
python code/download_data.py

# Ou séparément
python code/download_data.py --camvid   # CamVid uniquement
python code/download_data.py --sam      # Poids SAM uniquement (~2.5GB)
python code/download_data.py --coco     # Annotations COCO uniquement (~241MB)
```

> Les images COCO (4000 images urbaines ~400MB) sont téléchargées
> automatiquement à la demande lors de l'exécution de projet_half_open.py.

## Lancement sur cluster SLURM
```bash
# Vocabulaire fermé
sbatch scripts/run_closed.sh

# Vocabulaire semi-ouvert
sbatch scripts/run_halfopen.sh
```

## Fichiers cache

Les fichiers `.pkl` et `.pth` sont générés automatiquement au premier lancement.

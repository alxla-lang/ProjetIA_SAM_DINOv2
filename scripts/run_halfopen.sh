#!/usr/bin/env bash
#SBATCH --job-name=SAM_DINOv2_HalfOpen
#SBATCH --partition=ENSTA-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/halfopen_%j.out
#SBATCH --error=logs/halfopen_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate projet_ia

echo "=== Téléchargement des données ==="
python code/download_data.py --camvid --sam --coco

echo "=== Lancement pipeline half-open vocabulary ==="
python code/projet_half_open.py

echo "=== Terminé ==="

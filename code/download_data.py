# download_data.py
# Télécharge CamVid, les poids SAM et les annotations COCO
import os
import subprocess
import zipfile
import argparse

def download_camvid():
    if not os.path.exists("CamVid"):
        print("Téléchargement CamVid...")
        subprocess.run([
            "git", "clone",
            "https://github.com/lih627/CamVid.git"
        ], check=True)
        print("CamVid téléchargé.")
    else:
        print("CamVid déjà présent.")

def download_sam():
    ckpt = "sam_vit_h_4b8939.pth"
    if not os.path.exists(ckpt):
        print("Téléchargement poids SAM (~2.5GB)...")
        subprocess.run([
            "wget", "-q",
            "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        ], check=True)
        print("Poids SAM téléchargés.")
    else:
        print("Poids SAM déjà présents.")

def download_coco_annotations():
    ann_path = os.path.join("coco2017", "annotations",
                            "instances_train2017.json")
    if not os.path.exists(ann_path):
        os.makedirs(os.path.join("coco2017", "annotations"), exist_ok=True)
        os.makedirs(os.path.join("coco2017", "images", "train2017"),
                    exist_ok=True)
        zip_path = os.path.join("coco2017", "annotations_trainval2017.zip")
        print("Téléchargement annotations COCO (~241MB)...")
        subprocess.run([
            "wget", "-q", "-O", zip_path,
            "http://images.cocodataset.org/annotations/"
            "annotations_trainval2017.zip"
        ], check=True)
        print("Extraction...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall("coco2017")
        os.remove(zip_path)
        print("Annotations COCO extraites.")
    else:
        print("Annotations COCO déjà présentes.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Téléchargement des données pour le pipeline SAM+DINOv2"
    )
    parser.add_argument("--camvid", action="store_true")
    parser.add_argument("--sam",    action="store_true")
    parser.add_argument("--coco",   action="store_true")
    args = parser.parse_args()

    # Si aucun flag → tout télécharger
    all_dl = not (args.camvid or args.sam or args.coco)

    if all_dl or args.camvid:
        download_camvid()
    if all_dl or args.sam:
        download_sam()
    if all_dl or args.coco:
        download_coco_annotations()

    print("\nTéléchargements terminés.")
    print("Les images COCO seront téléchargées à la demande lors de"
          " l'exécution de projet_half_open.py.")
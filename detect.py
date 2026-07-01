"""Détection YOLO OBB + calcul d'un point cible à distance fixe du centre du 'rond'.

Usage :
    python detect.py                                  (ouvre des fenêtres de sélection)
    python detect.py chemin/vers/best.pt               (ouvre une fenêtre pour l'image)
    python detect.py chemin/vers/best.pt image.jpg      (aucune fenêtre)
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO


def pick_file(title, filetypes):
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    return path


def parse_args():
    parser = argparse.ArgumentParser(description="Détection YOLO OBB + calcul du point cible")
    parser.add_argument("model", nargs="?", help="Chemin vers le modèle (best.pt). Si omis, une fenêtre s'ouvre pour le choisir.")
    parser.add_argument("image", nargs="?", help="Chemin vers l'image à tester. Si omis, une fenêtre s'ouvre pour la choisir.")
    parser.add_argument("--dist", type=float, default=90.0, help="Distance du point depuis le centre du rond, vers la puce (px)")
    parser.add_argument("--conf", type=float, default=0.25, help="Seuil de confiance")
    parser.add_argument("--imgsz", type=int, default=1024, help="Taille d'image pour l'inférence")
    parser.add_argument("--output", help="Chemin de sauvegarde de l'image annotée (par défaut : <image>_annotated.<ext>)")
    parser.add_argument("--no-show", action="store_true", help="Ne pas ouvrir de fenêtre d'affichage, juste sauvegarder")
    return parser.parse_args()


def main():
    args = parse_args()

    model_str = args.model or pick_file("Sélectionner le modèle (.pt)", [("Modèle YOLO", "*.pt")])
    if not model_str:
        sys.exit("Aucun modèle sélectionné.")
    model_path = Path(model_str)

    image_str = args.image or pick_file("Sélectionner une image", [("Images", "*.jpg *.jpeg *.png *.bmp")])
    if not image_str:
        sys.exit("Aucune image sélectionnée.")
    img_path = Path(image_str)

    if not model_path.exists():
        sys.exit(f"Modèle introuvable : {model_path}")
    if not img_path.exists():
        sys.exit(f"Image introuvable : {img_path}")

    model = YOLO(str(model_path))
    print(f"Modèle chargé : {model_path}")

    r = model.predict(str(img_path), imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
    img = r.plot(labels=False, conf=False)
    H, W = img.shape[:2]

    centres, scores = {}, {}
    if r.obb is not None and len(r.obb):
        xywhr = r.obb.xywhr.cpu().numpy()
        cls = r.obb.cls.cpu().numpy()
        conf = r.obb.conf.cpu().numpy()
        best = {}
        for i in range(len(xywhr)):
            name = model.names[int(cls[i])]
            if name not in best or conf[i] > best[name]:
                best[name] = conf[i]
                centres[name] = xywhr[i, :2]
                scores[name] = float(conf[i])

    if "bas-de-puce" in centres and "rond" in centres:
        p_puce, p_rond = centres["bas-de-puce"], centres["rond"]
        u = (p_puce - p_rond) / np.linalg.norm(p_puce - p_rond)
        point = p_rond + args.dist * u
        x, y = int(round(point[0])), int(round(point[1]))
        cv2.circle(img, (x, y), 8, (0, 0, 255), -1)

        lines = [
            f"Point : ({x}, {y}) px",
            f"bas-de-puce : {scores['bas-de-puce']:.2f}",
            f"rond : {scores['rond']:.2f}",
        ]
        FONT, SCALE, TH = cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3
        margin, gap = 25, 20
        sizes = [cv2.getTextSize(t, FONT, SCALE, TH)[0] for t in lines]
        line_h = max(s[1] for s in sizes) + gap
        block_w = max(s[0] for s in sizes)
        y0 = H - margin - line_h * (len(lines) - 1)
        overlay = img.copy()
        cv2.rectangle(overlay, (W - block_w - 2 * margin, y0 - line_h), (W, H), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.45, img, 0.55, 0)
        for i, t in enumerate(lines):
            px = W - sizes[i][0] - margin
            py = y0 + i * line_h
            cv2.putText(img, t, (px, py), FONT, SCALE, (0, 0, 0), TH + 3)
            cv2.putText(img, t, (px, py), FONT, SCALE, (0, 255, 255), TH)

        print(f"Point : x={x}, y={y} px | bas-de-puce: {scores['bas-de-puce']:.2f} | rond: {scores['rond']:.2f}")
    else:
        print("Il faut détecter 'bas-de-puce' ET 'rond'. Détecté :", list(centres.keys()))

    output_path = Path(args.output) if args.output else img_path.with_name(f"{img_path.stem}_annotated{img_path.suffix}")
    cv2.imwrite(str(output_path), img)
    print(f"Image annotée sauvegardée : {output_path}")

    if not args.no_show:
        plt.figure(figsize=(12, 12))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()

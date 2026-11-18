"""
Inferencia completa: imágenes TEM (uint8, 4096×3008, 2,152 nm/px) -> instancias uint16 (axón X par,
mielina X+1) con el mismo nombre de archivo.

Pipeline (idéntico al validado en el fold 0):
    1. submuestreo ×2 por media de bloques 2×2 (como en el entrenamiento, ~4,3 nm/px)
    2. nnU-Net 2D: probabilidades (fondo, axón, mielina), con espejado (TTA) y, si se dan varios
       folds, promedio de las probabilidades de todos ellos (ensemble); con varias carpetas -m, promedio
       con el mismo peso por carpeta (entrega final: 5 folds × 250 épocas + 5 folds × 500 épocas)
    3. reescalado bilineal de las probabilidades a la resolución nativa, P(mielina) × --myelin-weight
       (1,0 por defecto) y argmax
    4. watershed desde los axones -> una fibra por axón (X / X+1)
    5. filtros: fibras cuyo AXÓN toca el borde de la imagen y fibras sin mielina se eliminan
       (criterio de anotación del challenge, ver baseline/postprocess.py)
    6. nunca se entrega una imagen vacía: si los filtros borran todo, se conserva el paso 4

Uso:
    python predict.py -i TEST_DIR -o OUT_DIR -m MODEL_DIR [MODEL_DIR ...] [-f 0 1 2 3 4 | -f all]
MODEL_DIR = carpeta de resultados de nnU-Net (contiene plans.json, dataset.json y fold_X/).
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent / "baseline"))
from postprocess import filter_fibers, reweight_probs, semantic_to_instances  # noqa: E402

SCALE = 2
EXTS = (".tif", ".tiff", ".png")


def read_image(p: Path) -> np.ndarray:
    img = tifffile.imread(p) if p.suffix.lower() in (".tif", ".tiff") else np.asarray(Image.open(p))
    if img.ndim == 3:  # por si viniera con canales repetidos
        img = img[..., 0]
    return img


def write_label(p: Path, lbl: np.ndarray):
    lbl = lbl.astype(np.uint16)
    if p.suffix.lower() in (".tif", ".tiff"):
        tifffile.imwrite(p, lbl)
    else:
        Image.fromarray(lbl).save(p)


def downsample(img: np.ndarray, s: int = SCALE) -> np.ndarray:
    """Media de bloques s×s redondeada a uint8 (igual que baseline/convert_to_nnunet.py)."""
    H, W = img.shape
    b = img[: H - H % s, : W - W % s].reshape(H // s, s, W // s, s).astype(np.float32)
    return np.round(b.mean(axis=(1, 3))).astype(np.uint8)


def probs_to_label(probs: np.ndarray, shape, device, filt: dict, myelin_weight: float = 1.0) -> np.ndarray:
    p = torch.from_numpy(np.asarray(probs, dtype=np.float32)).to(device)
    if p.ndim == 4:
        p = p[:, 0]
    if tuple(p.shape[-2:]) != tuple(shape):
        p = F.interpolate(p[None], size=tuple(shape), mode="bilinear", align_corners=False)[0]
    sem = reweight_probs(p, myelin_weight).argmax(0).to(torch.uint8).cpu().numpy()
    inst = semantic_to_instances(sem)
    out = filter_fibers(inst, **filt)
    return out if out.any() else inst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", type=Path, required=True)
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("-m", "--model", type=Path, nargs="+", required=True,
                    help="una o varias carpetas de modelo; sus probabilidades se promedian con el mismo peso")
    ap.add_argument("-f", "--folds", nargs="+", default=["0", "1", "2", "3", "4"])
    ap.add_argument("-chk", "--checkpoint", default="checkpoint_final.pth")
    ap.add_argument("--step", type=float, default=0.5, help="solapamiento de ventanas de nnU-Net")
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--keep-border", action="store_true")
    ap.add_argument("--border-mode", choices=["axon", "fiber"], default="axon")
    ap.add_argument("--keep-axon-only", action="store_true")
    ap.add_argument("--myelin-weight", type=float, default=1.0, help="factor sobre P(mielina) antes del argmax")
    ap.add_argument("--min-circularity", type=float, default=0.0)
    ap.add_argument("--min-solidity", type=float, default=0.0)
    ap.add_argument("--cases", nargs="*", default=None, help="solo estos nombres (sin extensión)")
    a = ap.parse_args()

    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds = tuple(int(f) if f.isdigit() else f for f in a.folds)
    preds = []
    for m in a.model:
        pred = nnUNetPredictor(tile_step_size=a.step, use_mirroring=not a.no_mirror, device=device,
                               allow_tqdm=False)
        pred.initialize_from_trained_model_folder(str(m), folds, a.checkpoint)
        preds.append(pred)
    filt = dict(border_margin=None if a.keep_border else 0, drop_axon_only=not a.keep_axon_only,
                border_mode=a.border_mode, min_circularity=a.min_circularity, min_solidity=a.min_solidity)

    files = sorted(p for p in a.input.iterdir() if p.suffix.lower() in EXTS)
    if a.cases is not None:
        files = [p for p in files if p.stem in set(a.cases)]
    a.output.mkdir(parents=True, exist_ok=True)
    print(f"{len(files)} imágenes | {len(preds)} carpeta(s) de modelo × folds {folds} | {device}", flush=True)
    for i, p in enumerate(files, 1):
        img = read_image(p)
        small = downsample(img)[None, None].astype(np.float32)  # (c=1, 1, H, W) como NaturalImage2DIO
        probs = None
        for pred in preds:
            _, pr = pred.predict_single_npy_array(small, {"spacing": (999, 1, 1)}, None, None, True)
            probs = pr if probs is None else probs + pr
        probs = probs / len(preds)
        lbl = probs_to_label(probs, img.shape, device, filt, a.myelin_weight)
        write_label(a.output / p.name, lbl)
        print(f"[{i}/{len(files)}] {p.name}: {int((lbl >= 2).any() and lbl.max() // 2)} fibras", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("nnUNet_raw", "")
    os.environ.setdefault("nnUNet_preprocessed", "")
    os.environ.setdefault("nnUNet_results", "")
    main()

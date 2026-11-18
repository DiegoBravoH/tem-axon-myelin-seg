"""
Convierte el train del challenge al formato nnU-Net v2 (2D, ½ resolución, etiqueta ignore).

Etiquetas de instancia del challenge -> semántica nnU-Net:
    0                 -> 0 background
    X par >= 2        -> 1 axon
    X+1 impar >= 3    -> 2 myelin
    1 (artefacto)     -> 3 ignore   (nnU-Net exige que "ignore" sea la etiqueta más alta)

Submuestreo ×2 (4096×3008 -> 2048×1504):
    imagen   -> media de cada bloque 2×2 (centrada igual que el bloque, sin desplazamiento)
    etiqueta -> voto mayoritario en el bloque 2×2; empates: mielina > axón > fondo > ignore
                (prioriza la mielina, que es la estructura más fina)

Uso:
    python baseline/convert_to_nnunet.py [--workers N] [--scale 2]
Requiere la variable de entorno nnUNet_raw (por defecto <repo>/nnUNet_raw).
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exploratory_analysis"))
from common import IMG_DIR, LBL_DIR, ROOT  # noqa: E402

# variante "ignore" (501): artefacto = ignore.  variante "artifact" (502): artefacto = 4.ª clase entrenable
VARIANTS = {
    "ignore": (501, "Dataset501_TEMAxonMyelin", {"background": 0, "axon": 1, "myelin": 2, "ignore": 3}),
    "artifact": (502, "Dataset502_TEMAxonMyelinArt", {"background": 0, "axon": 1, "myelin": 2, "artifact": 3}),
}
LABELS = VARIANTS["ignore"][2]
TIE_PRIORITY = [2, 1, 0, 3]  # se redefine en main(): mielina > axón > (ignore|artefacto) > fondo


def to_semantic(inst: np.ndarray) -> np.ndarray:
    sem = np.zeros(inst.shape, dtype=np.uint8)
    sem[inst == 1] = 3  # ignore (501) o artefacto (502)
    sem[(inst >= 2) & (inst % 2 == 0)] = 1
    sem[(inst >= 3) & (inst % 2 == 1)] = 2
    return sem


def downsample_image(img: np.ndarray, s: int) -> np.ndarray:
    H, W = img.shape
    blocks = img[: H - H % s, : W - W % s].reshape(H // s, s, W // s, s).astype(np.float32)
    return np.round(blocks.mean(axis=(1, 3))).astype(np.uint8)


def downsample_labels(sem: np.ndarray, s: int, priority=TIE_PRIORITY) -> np.ndarray:
    H, W = sem.shape
    blocks = sem[: H - H % s, : W - W % s].reshape(H // s, s, W // s, s)
    # conteo por clase en el orden de prioridad; argmax devuelve el primero en caso de empate
    counts = np.stack([(blocks == c).sum(axis=(1, 3)) for c in priority])
    return np.asarray(priority, dtype=np.uint8)[counts.argmax(axis=0)]


def convert(args):
    sample_id, out_img, out_lbl, s, priority = args
    img = tifffile.imread(IMG_DIR / f"{sample_id}.tif")
    sem = to_semantic(tifffile.imread(LBL_DIR / f"{sample_id}.tif"))
    if s > 1:
        img, sem = downsample_image(img, s), downsample_labels(sem, s, priority)
    Image.fromarray(img).save(out_img / f"{sample_id}_0000.png")
    Image.fromarray(sem).save(out_lbl / f"{sample_id}.png")
    return sample_id, np.bincount(sem.ravel(), minlength=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--variant", choices=list(VARIANTS), default="ignore")
    a = ap.parse_args()
    _, DATASET_NAME, LABELS = VARIANTS[a.variant]
    # empates: mielina > axón > fondo > ignore  |  mielina > axón > artefacto > fondo
    priority = [2, 1, 0, 3] if a.variant == "ignore" else [2, 1, 3, 0]

    raw = Path(os.environ.get("nnUNet_raw", ROOT / "nnUNet_raw"))
    out = raw / DATASET_NAME
    out_img, out_lbl = out / "imagesTr", out / "labelsTr"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    ids = sorted(p.stem for p in IMG_DIR.glob("*.tif"))
    missing = [i for i in ids if not (LBL_DIR / f"{i}.tif").exists()]
    if missing:
        raise SystemExit(f"Faltan etiquetas para: {missing}")

    with ProcessPoolExecutor(a.workers) as ex:
        results = list(ex.map(convert, [(i, out_img, out_lbl, a.scale, priority) for i in ids]))

    total = sum(c for _, c in results)
    (out / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "TEM"},
        "labels": LABELS,
        "numTraining": len(ids),
        "file_ending": ".png",
        "overwrite_image_reader_writer": "NaturalImage2DIO",
        "description": f"SIPAIM 2026 TEM axon/myelin, train a 1/{a.scale} de resolución",
    }, indent=2))
    frac = total / total.sum()
    print(f"{len(ids)} casos en {out}")
    print("fracción de píxeles:", {k: round(float(frac[v]), 4) for k, v in LABELS.items()})


if __name__ == "__main__":
    main()

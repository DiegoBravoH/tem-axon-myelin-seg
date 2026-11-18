"""
Evaluador local con las métricas oficiales del challenge (content/submission.md).

Por imagen, tras excluir los píxeles con GT == 1 (artefactos) en GT y predicción:
    Dice      axón (pares >= 2) y mielina (impares >= 3), y su media
    HD95 (nm) contornos de la máscara de axón y de mielina; máx. de los dos P95 dirigidos × 2,152
    MAE_g     fibras X ∪ X+1 emparejadas por IoU; g = sqrt(A_axón / A_fibra)
Extra (no oficial): precisión / recall / F1 de detección de fibras a IoU >= --iou.

Decisiones donde el protocolo oficial es ambiguo (configurables). El resumen informa también las variantes:
    dice_*_pooled        Dice sumando píxeles de todas las imágenes (en vez de media por imagen)
    mae_g_pooled_anyIoU  emparejamiento por máximo IoU sin umbral (cualquier solapamiento)

    - emparejamiento uno a uno, por IoU descendente, con IoU >= --iou (0,5 por defecto)
    - g-ratio solo para fibras del GT con axón y mielina (pares completos)
    - HD95 = NaN si una de las dos máscaras está vacía (se informa cuántas)
    - contorno = píxeles de la máscara con algún vecino-4 fuera de ella (el borde de la imagen no cuenta)

Uso:
    python baseline/evaluate.py PRED_DIR [--gt-dir DIR] [--out CSV] [--workers N]
Evalúa solo los archivos .tif/.png presentes en PRED_DIR (útil para un fold de validación).
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exploratory_analysis"))
from common import LBL_DIR, PIXEL_SIZE_NM  # noqa: E402

CROSS = ndi.generate_binary_structure(2, 1)


def read_label(path: Path) -> np.ndarray:
    if path.suffix.lower() in (".tif", ".tiff"):
        return tifffile.imread(path)
    return np.asarray(Image.open(path))


def dice(p: np.ndarray, g: np.ndarray) -> float:
    s = p.sum() + g.sum()
    return 1.0 if s == 0 else 2.0 * (p & g).sum() / s


def contour(m: np.ndarray) -> np.ndarray:
    return m & ~ndi.binary_erosion(m, CROSS, border_value=1)


def hd95(p: np.ndarray, g: np.ndarray, px: float) -> float:
    if not p.any() and not g.any():
        return 0.0
    if not p.any() or not g.any():
        return np.nan
    cp, cg = contour(p), contour(g)
    d_to_g = ndi.distance_transform_edt(~cg)[cp]
    d_to_p = ndi.distance_transform_edt(~cp)[cg]
    return max(np.percentile(d_to_g, 95), np.percentile(d_to_p, 95)) * px


def fiber_table(lbl: np.ndarray) -> pd.DataFrame:
    """Áreas por fibra (id = X // 2)."""
    fid = np.where(lbl >= 2, lbl // 2, 0).astype(np.int64)
    n = int(fid.max()) + 1
    a_ax = np.bincount(fid[(lbl >= 2) & (lbl % 2 == 0)], minlength=n)
    a_fib = np.bincount(fid.ravel(), minlength=n)
    ids = np.nonzero(a_fib)[0]
    ids = ids[ids > 0]
    return pd.DataFrame({"fid": ids, "a_axon": a_ax[ids], "a_fiber": a_fib[ids]}).set_index("fid")


def match_fibers(pred: np.ndarray, gt: np.ndarray, iou_thr: float):
    """Emparejamiento uno a uno por IoU descendente. Devuelve (pares, tabla_pred, tabla_gt)."""
    fp = np.where(pred >= 2, pred // 2, 0).astype(np.int64)
    fg = np.where(gt >= 2, gt // 2, 0).astype(np.int64)
    tp, tg = fiber_table(pred), fiber_table(gt)
    both = (fp > 0) & (fg > 0)
    if not both.any():
        return [], tp, tg
    code = fg[both] * (fp.max() + 1) + fp[both]
    codes, inter = np.unique(code, return_counts=True)
    g_ids, p_ids = codes // (fp.max() + 1), codes % (fp.max() + 1)
    union = tg.loc[g_ids, "a_fiber"].to_numpy() + tp.loc[p_ids, "a_fiber"].to_numpy() - inter
    iou = inter / union
    order = np.argsort(-iou)
    used_g, used_p, pairs = set(), set(), []
    for k in order:
        if iou[k] < iou_thr:
            break
        g, p = int(g_ids[k]), int(p_ids[k])
        if g in used_g or p in used_p:
            continue
        used_g.add(g); used_p.add(p)
        pairs.append((g, p, float(iou[k])))
    return pairs, tp, tg


def evaluate_case(args):
    name, pred_path, gt_path, iou_thr, px = args
    gt = read_label(gt_path).astype(np.int64)
    pred = read_label(pred_path).astype(np.int64)
    if pred.shape != gt.shape:
        raise ValueError(f"{name}: forma pred {pred.shape} != GT {gt.shape}")
    ignore = gt == 1
    gt = np.where(ignore, 0, gt)
    pred = np.where(ignore | (pred == 1), 0, pred)

    row = {"case": name}
    for cls, sel in (("axon", lambda x: (x >= 2) & (x % 2 == 0)), ("myelin", lambda x: (x >= 3) & (x % 2 == 1))):
        p, g = sel(pred), sel(gt)
        row[f"dice_{cls}"] = dice(p, g)
        row[f"_inter_{cls}"], row[f"_sum_{cls}"] = int((p & g).sum()), int(p.sum() + g.sum())
        row[f"hd95_{cls}_nm"] = hd95(p, g, px)
    row["dice_mean"] = (row["dice_axon"] + row["dice_myelin"]) / 2
    row["hd95_mean_nm"] = (row["hd95_axon_nm"] + row["hd95_myelin_nm"]) / 2

    pairs, tp, tg = match_fibers(pred, gt, iou_thr)
    complete_gt = tg[(tg["a_axon"] > 0) & (tg["a_axon"] < tg["a_fiber"])].index
    errs = [abs(np.sqrt(tp.at[p, "a_axon"] / tp.at[p, "a_fiber"]) - np.sqrt(tg.at[g, "a_axon"] / tg.at[g, "a_fiber"]))
            for g, p, _ in pairs if g in complete_gt]
    row["mae_g"] = float(np.mean(errs)) if errs else np.nan
    row["n_matched_g"] = len(errs)
    pairs0, _, _ = match_fibers(pred, gt, 1e-9)
    errs0 = [abs(np.sqrt(tp.at[p, "a_axon"] / tp.at[p, "a_fiber"]) - np.sqrt(tg.at[g, "a_axon"] / tg.at[g, "a_fiber"]))
             for g, p, _ in pairs0 if g in complete_gt]
    row["_mae_g_any_sum"], row["_n_any"] = float(np.sum(errs0)), len(errs0)
    n_tp, n_pred, n_gt = len(pairs), len(tp), len(tg)
    row["det_precision"] = n_tp / n_pred if n_pred else np.nan
    row["det_recall"] = n_tp / n_gt if n_gt else np.nan
    row["det_f1"] = 2 * n_tp / (n_pred + n_gt) if (n_pred + n_gt) else np.nan
    row["n_pred_fibers"], row["n_gt_fibers"] = n_pred, n_gt
    return row


def summarize(df: pd.DataFrame) -> pd.Series:
    cols = ["dice_axon", "dice_myelin", "dice_mean", "hd95_axon_nm", "hd95_myelin_nm", "hd95_mean_nm",
            "mae_g", "det_precision", "det_recall", "det_f1"]
    s = df[cols].mean()
    # MAE_g oficial: media sobre TODAS las fibras emparejadas (no media de medias por imagen)
    w = df["n_matched_g"]
    s["mae_g_pooled"] = float((df["mae_g"].fillna(0) * w).sum() / w.sum()) if w.sum() else np.nan
    for c in ("axon", "myelin"):
        s[f"dice_{c}_pooled"] = 2 * df[f"_inter_{c}"].sum() / df[f"_sum_{c}"].sum()
    s["dice_mean_pooled"] = (s["dice_axon_pooled"] + s["dice_myelin_pooled"]) / 2
    s["mae_g_pooled_anyIoU"] = df["_mae_g_any_sum"].sum() / max(df["_n_any"].sum(), 1)
    s["n_hd95_nan"] = int(df[["hd95_axon_nm", "hd95_myelin_nm"]].isna().any(axis=1).sum())
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred_dir", type=Path)
    ap.add_argument("--gt-dir", type=Path, default=LBL_DIR)
    ap.add_argument("--out", type=Path, default=None, help="CSV por imagen (por defecto PRED_DIR/metrics.csv)")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    preds = sorted(p for p in a.pred_dir.iterdir() if p.suffix.lower() in (".tif", ".tiff", ".png"))
    jobs = []
    for p in preds:
        g = next((a.gt_dir / f"{p.stem}{e}" for e in (".tif", ".png") if (a.gt_dir / f"{p.stem}{e}").exists()), None)
        if g is None:
            raise SystemExit(f"Sin GT para {p.name}")
        jobs.append((p.stem, p, g, a.iou, PIXEL_SIZE_NM))
    if not jobs:
        raise SystemExit(f"No hay predicciones en {a.pred_dir}")

    with ProcessPoolExecutor(a.workers) as ex:
        df = pd.DataFrame(ex.map(evaluate_case, jobs))
    out = a.out or a.pred_dir / "metrics.csv"
    df.to_csv(out, index=False)
    s = summarize(df)
    print(f"{len(df)} imágenes  ->  {out}")
    print(s.round(4).to_string())


if __name__ == "__main__":
    main()

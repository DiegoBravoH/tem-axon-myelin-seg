"""
Probabilidades de nnU-Net (.npz, a resolución reducida) -> instancias uint16 a resolución nativa.

    1. reescala las probabilidades (bilineal, align_corners=False: coincide con la media 2×2 usada
       al submuestrear) al tamaño nativo 3008×4096 y hace argmax (en GPU si hay)
    2. watershed (postprocess.semantic_to_instances) -> formato X / X+1
    3. filtros (postprocess.filter_fibers): por defecto se borran las fibras que tocan el borde
       y las fibras sin mielina (validado en fold 0, zero-shot: HD95 951 -> 722 nm, Dice 0,765 -> 0,790)

Uso:
    python baseline/probs_to_instances.py RAW_DIR OUT_DIR [--border-margin 0 | --keep-border]
           [--keep-axon-only] [--myelin-weight W] [--min-conf C] [--min-fiber-px N] [--min-axon-px N] [--workers 8] [--cases a b ...]
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from postprocess import filter_fibers, reweight_probs, semantic_to_instances  # noqa: E402


def upsampled_argmax(probs: np.ndarray, shape, device, return_fg: bool = False, myelin_weight: float = 1.0):
    """argmax a resolución nativa; con return_fg devuelve también P(axón)+P(mielina) (float16)."""
    p = torch.from_numpy(probs.astype(np.float32)).to(device)
    if p.ndim == 4:  # (C, 1, H, W) en datos 2D de nnU-Net
        p = p[:, 0]
    if tuple(p.shape[-2:]) != tuple(shape):
        p = F.interpolate(p[None], size=tuple(shape), mode="bilinear", align_corners=False)[0]
    sem = reweight_probs(p.clone() if return_fg else p, myelin_weight).argmax(0).to(torch.uint8).cpu().numpy()
    if not return_fg:
        return sem
    return sem, (p[1] + p[2]).to(torch.float16).cpu().numpy()


def _post(args):
    sem, fg, out_path, kw = args
    inst = semantic_to_instances(sem, kw["min_axon_px"])
    inst = filter_fibers(inst, kw["border_margin"], kw["drop_axon_only"], kw["min_fiber_px"], fg, kw["min_conf"],
                         kw["border_mode"], kw["min_solidity"], kw["min_circularity"])
    tifffile.imwrite(out_path, inst)
    return out_path.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--shape", type=int, nargs=2, default=(3008, 4096))
    ap.add_argument("--border-margin", type=int, default=0)
    ap.add_argument("--keep-border", action="store_true", help="no borrar fibras del borde")
    ap.add_argument("--border-mode", choices=["axon", "fiber"], default="axon",
                    help="axon: borrar solo si el axón toca el borde (por defecto); fiber: si cualquier píxel lo toca")
    ap.add_argument("--keep-axon-only", action="store_true", help="no borrar fibras sin mielina")
    ap.add_argument("--min-fiber-px", type=int, default=0)
    ap.add_argument("--min-axon-px", type=int, default=0)
    ap.add_argument("--min-solidity", type=float, default=0.0)
    ap.add_argument("--min-circularity", type=float, default=0.0)
    ap.add_argument("--min-conf", type=float, default=0.0, help="confianza media mínima por fibra (0 = sin filtro)")
    ap.add_argument("--myelin-weight", type=float, default=1.0, help="factor sobre P(mielina) antes del argmax")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cases", nargs="*", default=None, help="solo estos casos (sin extensión)")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    kw = dict(min_axon_px=a.min_axon_px, border_margin=None if a.keep_border else a.border_margin,
              drop_axon_only=not a.keep_axon_only, min_fiber_px=a.min_fiber_px, min_conf=a.min_conf,
              border_mode=a.border_mode, min_solidity=a.min_solidity, min_circularity=a.min_circularity)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    files = sorted(a.raw_dir.glob("*.npz"))
    if a.cases is not None:
        files = [f for f in files if f.stem in set(a.cases)]
    with ProcessPoolExecutor(a.workers) as ex:
        futs = []
        for f in files:
            if a.min_conf > 0:
                sem, fg = upsampled_argmax(np.load(f)["probabilities"], a.shape, device, return_fg=True,
                                           myelin_weight=a.myelin_weight)
            else:
                sem, fg = upsampled_argmax(np.load(f)["probabilities"], a.shape, device,
                                           myelin_weight=a.myelin_weight), None
            futs.append(ex.submit(_post, (sem, fg, a.out_dir / f"{f.stem}.tif", kw)))
        for i, fu in enumerate(futs, 1):
            fu.result()
            if i % 25 == 0 or i == len(futs):
                print(f"{i}/{len(futs)}", flush=True)


if __name__ == "__main__":
    main()

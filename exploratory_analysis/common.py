"""Rutas y constantes compartidas por los scripts del proyecto.

La ruta de los datos puede sobrescribirse con la variable de entorno SIPAIM_DATA_DIR.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("SIPAIM_DATA_DIR", ROOT / "data" / "train-dataset" / "train-dataset"))
IMG_DIR = DATA_DIR / "train_samples"
LBL_DIR = DATA_DIR / "train_instance_labels"

PIXEL_SIZE_NM = 2.152  # resolución nativa del challenge (nm/px)


def nm_to_px(nm: float, pixel_size_nm: float = PIXEL_SIZE_NM) -> int:
    """Convierte una longitud física a píxeles a la resolución dada (mínimo 1 px)."""
    return max(1, round(nm / pixel_size_nm))

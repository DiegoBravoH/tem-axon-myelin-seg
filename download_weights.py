"""
Descarga los pesos entrenados (10 modelos nnU-Net: 5 folds × 250 épocas + 5 folds × 500 épocas) desde
Google Drive a checkpoints/ y verifica cada archivo con su SHA-256.

Resultado:
    checkpoints/model/        (250 épocas)  plans.json, dataset.json, fold_0..4/checkpoint_final.pth
    checkpoints/model_500ep/  (500 épocas)  ídem

Uso:
    python download_weights.py [--out checkpoints]
Requiere: pip install gdown
"""
import argparse
import hashlib
import sys
from pathlib import Path

DRIVE_FOLDER = "https://drive.google.com/drive/folders/1MJeqivSnRzy8AH8csZnSfNmfoHEv7Jee"

SHA256 = {
    "model/plans.json": "a13212241ad1a564ac711c1a10e15116c32356006e605ca33a249513d8385db1",
    "model/dataset.json": "8b104f7a63340c1e98735c5f39823c8868c34737b73f27f7f8db18ff02417739",
    "model/fold_0/checkpoint_final.pth": "ed1dcb61d6f96f543a0adc6b4c3edbe4b8fe3ec2c5f0fc92277600e3dd60899b",
    "model/fold_1/checkpoint_final.pth": "7000bc2da5b9cd420e7c204bc44782177ba960ed8de1e3c54ecd554304ed96a0",
    "model/fold_2/checkpoint_final.pth": "f29bd95e976365ea97e451288f71b98512224e6a4481e845d59dbd7ea284ea67",
    "model/fold_3/checkpoint_final.pth": "c8f93e9d396cd225c6ac74f00ddec17d35e946638691c2a71c3991773da2d4c0",
    "model/fold_4/checkpoint_final.pth": "ccb59ad94967e04694dd9753cd5e316ff7ef478c63a31c13bfbaf486034cd896",
    "model_500ep/plans.json": "a13212241ad1a564ac711c1a10e15116c32356006e605ca33a249513d8385db1",
    "model_500ep/dataset.json": "8b104f7a63340c1e98735c5f39823c8868c34737b73f27f7f8db18ff02417739",
    "model_500ep/fold_0/checkpoint_final.pth": "9a5ea641859c51b5980b637c7e93344fd5e2de510eeca582b7b08a0a2e64764f",
    "model_500ep/fold_1/checkpoint_final.pth": "5e0b7af4c9596e224469cc0bfc4c6e7548263e4540b5ecf277c7cf9754eea7f5",
    "model_500ep/fold_2/checkpoint_final.pth": "e8ef6717eca319793b52d4228d559e352b97ff68074183a080dc0dd72c4cda3e",
    "model_500ep/fold_3/checkpoint_final.pth": "85b3f2764459b3524da0cf2098d23a1af02b1076bb9ed8f84f17fe6f6052295d",
    "model_500ep/fold_4/checkpoint_final.pth": "d838aa23223b81d8d8cefbe7c2d50f8599eceb6782c197dde147120b9e9b2c72",
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(out: Path) -> list[str]:
    """Devuelve los archivos que faltan o cuyo SHA-256 no coincide."""
    bad = []
    for rel, h in SHA256.items():
        p = out / rel
        if not p.is_file():
            bad.append(f"falta      {rel}")
        elif sha256(p) != h:
            bad.append(f"corrupto   {rel}")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "checkpoints")
    a = ap.parse_args()

    if not verify(a.out):
        print(f"los pesos ya están en {a.out} y son correctos")
        return
    try:
        import gdown
    except ImportError:
        sys.exit("falta gdown: pip install gdown")
    print(f"descargando ~1,8 GB desde Google Drive a {a.out} ...")
    a.out.mkdir(parents=True, exist_ok=True)
    gdown.download_folder(DRIVE_FOLDER, output=str(a.out), quiet=False)
    bad = verify(a.out)
    if bad:
        sys.exit("la descarga no está completa:\n  " + "\n  ".join(bad) +
                 f"\nvuelve a ejecutar el script o descarga la carpeta a mano: {DRIVE_FOLDER}")
    print(f"OK: los 14 archivos están en {a.out} y su SHA-256 es correcto")


if __name__ == "__main__":
    main()

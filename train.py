"""
Entrenamiento completo, de los datos del challenge a los pesos usados en la entrega.

Pasos (cada uno se salta si su salida ya existe):
    1. descarga los pesos preentrenados multires-TEM de AxonDeepSeg (nnU-Net v2, TEM, axón/mielina)
    2. convierte el train al formato nnU-Net: ½ resolución (media 2×2 -> ~4,30 nm/px),
       etiquetas 0 fondo / 1 axón / 2 mielina / 3 ignore (= artefacto del GT, excluido de la pérdida)
    3. huella del dataset + plans: los del modelo preentrenado (misma arquitectura, parche y normalización)
       con el nombre del dataset cambiado
    4. preprocesado 2D de nnU-Net
    5. partición en 5 folds (generate_crossval_split, semilla 12345)
    6. fine-tuning de nnU-Net desde los pesos preentrenados, fold a fold, con 250 y con 500 épocas
       (nnUNetTrainer_250epochs y nnUNetTrainer_500epochs; la entrega promedia los 10 modelos)

Uso:
    python train.py --data-dir RUTA/train-dataset [--folds 0 1 2 3 4] [--epochs 250 500] [--gpu 0]
RUTA/train-dataset debe contener train_samples/ y train_instance_labels/ (sample_XXXX.tif).
Salida: nnUNet_results/Dataset501_TEMAxonMyelin/nnUNetTrainer_{250,500}epochs__nnUNetMultiresTEMPlans__2d/
        (las dos carpetas se pasan a predict.py con -m).
Tiempo: ~4 h (250 épocas) y ~8 h (500 épocas) por fold en una GPU de 24 GB (57 s/época).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent
DATASET_ID, DATASET = 501, "Dataset501_TEMAxonMyelin"
PLANS = "nnUNetMultiresTEMPlans"
TRAINER = "nnUNetTrainer_{}epochs"
PRETRAINED_URL = ("https://github.com/axondeepseg/model_seg_tem-multires/releases/download/v1.0/"
                  "model_seg_tem-multires_light.zip")
PRETRAINED_DIR = REPO / "pretrained" / "model_seg_tem-multires_light" / "model_seg_tem-multires_light"
SEED = 12345


def log(msg):
    print(f"[train] {msg}", flush=True)


def get_pretrained():
    if (PRETRAINED_DIR / "fold_all" / "checkpoint_final.pth").exists():
        return
    zdir = REPO / "pretrained" / "zips"
    zdir.mkdir(parents=True, exist_ok=True)
    zpath = zdir / PRETRAINED_URL.rsplit("/", 1)[1]
    if not zpath.exists():
        log(f"descargando {PRETRAINED_URL}")
        urllib.request.urlretrieve(PRETRAINED_URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(PRETRAINED_DIR.parent)


def convert():
    if (Path(os.environ["nnUNet_raw"]) / DATASET / "dataset.json").exists():
        return
    log("convirtiendo el dataset a formato nnU-Net (½ resolución)")
    subprocess.run([sys.executable, str(REPO / "baseline" / "convert_to_nnunet.py"), "--variant", "ignore"],
                   check=True)


def plan_and_preprocess():
    from nnunetv2.experiment_planning.plan_and_preprocess_api import extract_fingerprints, preprocess

    pre = Path(os.environ["nnUNet_preprocessed"]) / DATASET
    if not (pre / "dataset_fingerprint.json").exists():
        extract_fingerprints([DATASET_ID])
    pre.mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(os.environ["nnUNet_raw"]) / DATASET / "dataset.json", pre / "dataset.json")
    plans = json.loads((PRETRAINED_DIR / "plans.json").read_text())
    plans["dataset_name"] = DATASET
    plans["plans_name"] = PLANS
    plans["configurations"]["2d"]["data_identifier"] = f"{PLANS}_2d"
    (pre / f"{PLANS}.json").write_text(json.dumps(plans, indent=4))
    if not (pre / f"{PLANS}_2d").exists():
        log("preprocesando (2d)")
        preprocess([DATASET_ID], PLANS, configurations=("2d",), num_processes=(8,))
    if not (pre / "splits_final.json").exists():
        from nnunetv2.utilities.crossval_split import generate_crossval_split

        ids = sorted(p.name[: -len(".png")] for p in (Path(os.environ["nnUNet_raw"]) / DATASET / "labelsTr").glob("*.png"))
        (pre / "splits_final.json").write_text(json.dumps(generate_crossval_split(ids, seed=SEED, n_splits=5), indent=4))


def train(fold, epochs=250):
    import torch
    from nnunetv2.run.run_training import run_training

    trainer = TRAINER.format(epochs)
    out = Path(os.environ["nnUNet_results"]) / DATASET / f"{trainer}__{PLANS}__2d" / f"fold_{fold}"
    if (out / "checkpoint_final.pth").exists():
        log(f"fold {fold} ({epochs} épocas) ya entrenado")
        return
    log(f"entrenando fold {fold} ({epochs} épocas)")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    run_training(str(DATASET_ID), "2d", fold, trainer_class_name=trainer, plans_identifier=PLANS,
                 pretrained_weights=str(PRETRAINED_DIR / "fold_all" / "checkpoint_final.pth"),
                 export_validation_probabilities=True,
                 continue_training=(out / "checkpoint_latest.pth").exists(), device=torch.device("cuda"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True, help="carpeta con train_samples/ y train_instance_labels/")
    ap.add_argument("--folds", nargs="+", default=["0", "1", "2", "3", "4"], help="0..4 o 'all'")
    ap.add_argument("--epochs", nargs="+", type=int, default=[250, 500], choices=[250, 500])
    ap.add_argument("--gpu", default="0")
    a = ap.parse_args()

    os.environ["SIPAIM_DATA_DIR"] = str(a.data_dir.resolve())
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    for k in ("raw", "preprocessed", "results"):
        os.environ.setdefault(f"nnUNet_{k}", str(REPO / f"nnUNet_{k}"))
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(k, "1")

    get_pretrained()
    convert()
    plan_and_preprocess()
    for e in a.epochs:
        for f in a.folds:
            train(int(f) if f.isdigit() else f, e)


if __name__ == "__main__":
    main()

# Annotation-aware nnU-Net ensemble for axon and myelin instance segmentation in TEM

**2nd place** in the **SIPAIM 2026 TEM Axon & Myelin Segmentation Challenge**
([challenge website](https://alexandrojim.github.io/tem-challenge/)).

**Author:** Diego Fernando Bravo Higuera, Universidad Nacional de Colombia

## Task

Each myelinated fiber in a transmission electron microscopy (TEM) image of the mouse corpus callosum
(4096×3008 px, 2.152 nm/px) must be segmented as a linked pair of instances: an axon (`X`, even ≥ 2)
and its myelin sheath (`X+1`). Submissions are ranked by Dice, the 95th-percentile Hausdorff distance
(HD95) and the mean absolute error (MAE) of the per-fiber g-ratio.

## Method

1. **Preprocessing.** 2×2 block-mean downsampling (4096×3008 → 2048×1504 px, 2.15 → 4.30 nm/px) and
   per-image z-score normalization.
2. **Segmentation.** A 2D nnU-Net pretrained on TEM (AxonDeepSeg *multires-TEM*) is fine-tuned on the
   challenge data. Ten models (5 folds × 250/500 epochs) are ensembled, with mirroring test-time augmentation.
3. **Post-processing** at native resolution:
   - bilinear upsampling of the averaged probabilities (2048×1504 → 4096×3008 px);
   - myelin probability multiplied by 0.7 before the argmax (lowers the g-ratio error);
   - axon-seeded watershed: one fiber per axon, encoded as `X` / `X+1`;
   - annotation-aware filters: fibers whose axon touches the image border, or without myelin, are removed,
     following the annotation protocol.

## Cross-validation results

5-fold cross-validation on the 266 training images (out-of-fold predictions):

| Configuration | Dice (mean) | HD95 (nm) | g-ratio MAE |
|---|---|---|---|
| 250 epochs, no myelin weight | 0.8873 | 527 | 0.0101 |
| 250 epochs, myelin weight 0.7 | 0.8876 | 523 | 0.0096 |
| **250 + 500 epochs (final)** | **0.8889** | **497** | 0.0096 |

The metrics were computed with our own re-implementation of the official ones (`baseline/evaluate.py`).

## 1. Installation

```bash
git clone https://github.com/DiegoBravoH/tem-axon-myelin-seg.git
cd tem-axon-myelin-seg
conda create -n tem python=3.11 -y && conda activate tem
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```
Tested on Linux with a 24 GB NVIDIA GPU (CUDA 12.8). Inference also runs on CPU, but much more slowly.

## 2. Trained weights

The 10 trained models (~1.8 GB) are on
[Google Drive](https://drive.google.com/drive/folders/1MJeqivSnRzy8AH8csZnSfNmfoHEv7Jee).
Download them with:

```bash
python download_weights.py
```

The script places them in `checkpoints/` and checks the SHA-256 of every file:

```text
checkpoints/
├── model/          # 250 epochs
│   ├── plans.json
│   ├── dataset.json
│   └── fold_0 … fold_4/checkpoint_final.pth
└── model_500ep/    # 500 epochs, same structure
```

## 3. Inference

The test images are available on the [challenge website](https://alexandrojim.github.io/tem-challenge/)
(Dataset section). There are two equivalent ways to run inference. **Option A (`predict.py`) is the
recommended one:** it is the script that produced the submitted predictions. Option B (the notebook) is
optional and gives the same output.

### Option A (recommended): `predict.py`

```bash
python predict.py -i TEST_DIR -o predictions -m checkpoints/model checkpoints/model_500ep \
    -f 0 1 2 3 4 --myelin-weight 0.7
```

- `TEST_DIR`: test micrographs (`.tif`, uint8, 4096×3008, 2.152 nm/px).
- `checkpoints/model` (250 epochs) and `checkpoints/model_500ep` (500 epochs): nnU-Net results folders.
  Their probabilities are averaged with equal weight.
- Output: one `uint16` `.tif` per input, same file name. Values: 0 = background, `X` (even ≥ 2) = axon,
  `X+1` = its myelin sheath.

The pipeline in `predict.py`:
1. 2×2 block-mean downsampling (~4.30 nm/px, inside the pretrained model's resolution range).
2. nnU-Net sliding-window inference (step 0.5, mirroring TTA); softmax probabilities averaged over the 10 models.
   In 5-fold CV, adding the 500-epoch models lowered HD95 from 523 to 497 nm.
3. Bilinear upsampling of the probabilities to native resolution, then argmax (0 bg / 1 axon / 2 myelin)
   after scaling P(myelin) by 0.7 (`--myelin-weight`; in 5-fold CV this lowered the g-ratio MAE, with Dice and HD95 unchanged).
4. Instance separation: watershed on the distance to the nearest axon, seeded by the connected
   components of the axon mask and restricted to the axon ∪ myelin mask (one fiber per axon).
5. Filters that follow the annotation protocol: fibers whose **axon** touches the image border are
   removed (the ground truth marks them as artifact), and so are fibers with no myelin.

### Option B (optional): `inference.ipynb`

The notebook runs the same pipeline step by step, with the same parameters as the Option A command, and gives
the same masks (up to GPU non-determinism, about 1e-5 of the pixels).

```bash
pip install jupyterlab   # if Jupyter is not installed
jupyter lab inference.ipynb
```

Set `IMAGES_DIR` in the first code cell (relative to the repository folder, default `test_samples/`) and run
all cells. The notebook checks the image folder, the 14 model files and the libraries, then predicts each
image, saves its mask in `test_predictions/` and shows the image next to its prediction.

## 4. Re-train from scratch

```bash
python train.py --data-dir PATH/train-dataset --folds 0 1 2 3 4 --epochs 250 500 --gpu 0
```
`PATH/train-dataset` must contain `train_samples/` and `train_instance_labels/` (training data from the
[challenge website](https://alexandrojim.github.io/tem-challenge/)). The script downloads the pretrained weights
([model_seg_tem-multires v1.0](https://github.com/axondeepseg/model_seg_tem-multires/releases/tag/v1.0)),
converts the data (artifact label 1 → nnU-Net *ignore* label), reuses the pretrained plans, preprocesses,
writes the 5-fold split (seed 12345) and fine-tunes each fold for 250 epochs (~4 h per fold) and for 500 epochs
(~8 h per fold). The results land in
`nnUNet_results/Dataset501_TEMAxonMyelin/nnUNetTrainer_{250,500}epochs__nnUNetMultiresTEMPlans__2d/`;
pass both folders to `predict.py -m`.

## 5. Files

| File | Purpose |
|---|---|
| `download_weights.py` | downloads the trained weights from Google Drive and checks them |
| `predict.py` | inference + post-processing → submission masks |
| `inference.ipynb` | the same inference as a step-by-step notebook, with checks and figures |
| `train.py` | full training pipeline |
| `baseline/convert_to_nnunet.py` | challenge labels → nnU-Net semantic labels at ½ resolution |
| `baseline/postprocess.py` | watershed instance separation and fiber filters |
| `baseline/probs_to_instances.py` | the same post-processing applied to saved nnU-Net probabilities (validation) |
| `baseline/evaluate.py` | local reimplementation of the official metrics (Dice, HD95, g-ratio MAE) |
| `exploratory_analysis/common.py` | data paths and pixel size |

## Data

The data are **not** included in this repository. They can be downloaded from the
[challenge website](https://alexandrojim.github.io/tem-challenge/) (Dataset section).

## References

- F. Isensee et al., "nnU-Net: a self-configuring method for deep learning-based biomedical image
  segmentation," *Nature Methods*, 18:203–211, 2021.
- A. Zaimi et al., "AxonDeepSeg: automatic axon and myelin segmentation from microscopy data using
  convolutional neural networks," *Scientific Reports*, 8:3816, 2018.
- A. Collin, M. Boudreau, and J. Cohen-Adad, "ASTIH: a collection of axon and myelin segmentation datasets
  from multiple histology studies," *MIDL*, 2026.

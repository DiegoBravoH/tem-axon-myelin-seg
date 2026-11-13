# Annotation-aware nnU-Net ensemble for axon and myelin instance segmentation in TEM

Submission to the **SIPAIM 2026 TEM Axon & Myelin Segmentation Challenge**
([challenge website](https://alexandrojim.github.io/tem-challenge/)).

**Author:** Diego Fernando Bravo Higuera, Universidad Nacional de Colombia

> **Status:** the code, trained weights and example notebooks (local and Google Colab) will be added soon.

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

The metrics were computed with our own re-implementation of the official ones.

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

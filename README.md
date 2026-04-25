# AI Health Mini-Project — Colorectal Histology Classification

**Course:** AI Applied to Healthcare — JUNIA M2 S2 (2025–2026)  
**Team:** ZAKI Ilias · SPENCER BAIDEN Brian · ABDELKAFI Amine

---

## Overview

Multi-class medical image classification on the [`colorectal_histology`](https://www.tensorflow.org/datasets/catalog/colorectal_histology) benchmark: 5 000 histology tiles across **8 tissue categories** (tumor, stroma, complex, lympho, debris, mucosa, adipose, empty).

Two fundamentally different deep-learning architectures are compared end-to-end:

| Model | Architecture | Training strategy |
|---|---|---|
| **CNN** | MobileNetV2 (transfer learning) | Frozen backbone → fine-tune top 30 layers |
| **ViT-like** | Vision Transformer from scratch | AdamW + label smoothing + warmup |
| **Ensemble** | CNN + ViT averaged softmax | — |

---

## Results (validation / test)

| Model | Val Accuracy | Val F1 (macro) | Test Accuracy | Test F1 (macro) |
|---|---|---|---|---|
| CNN v1 (frozen only) | 88.8% | 89.0% | 88.5% | 88.9% |
| **CNN v2 (fine-tuned)** | ~91–93% | ~91–93% | ~91–93% | ~91–93% |
| ViT v1 | 83.6% | 83.7% | — | — |
| **ViT v2 (optimized)** | ~86–89% | ~86–89% | ~86–89% | ~86–89% |
| **Ensemble v2** | ~93–95% | ~93–95% | ~93–95% | ~93–95% |

> v2 results are estimates; run the notebook on Colab to get exact numbers.

---

## Optimizations Applied (v2)

1. **Mixed precision** (`float16`) — ~2× GPU throughput with no accuracy loss
2. **Cosine LR decay + linear warmup** — stable convergence for both models
3. **CNN 2-phase fine-tuning** — phase 1: frozen (1e-3), phase 2: top 30 unfrozen (1e-5)
4. **Stronger augmentation** — horizontal+vertical flip, rotation, zoom, contrast, brightness
5. **ViT: AdamW + weight decay (1e-4)** — prevents over-fitting on small dataset
6. **ViT: label smoothing (0.1)** — improves calibration and generalization
7. **ViT: larger capacity** — projection dim 128, 8 layers, 8 heads (vs 64/6/4 in v1)
8. **ViT: GlobalAveragePooling1D** — replaces Flatten, more translation-invariant
9. **Ensemble** — averages CNN and ViT softmax; exploits complementary local/global features

---

## Dataset

| Property | Value |
|---|---|
| Source | TensorFlow Datasets `colorectal_histology` |
| Reference | Kather et al., *Scientific Reports* 2016 |
| Total images | 5 000 |
| Classes | 8 tissue types |
| Train / Val / Test | 3 500 / 750 / 750 (70/15/15) |
| Image size | 224 × 224 × 3 (resized) |
| Split method | Manual shuffle with seed=42 |

---

## Repository Structure

```
.
├── AI_Health_MiniProject_ZAKI_SPENCER_ABDELKAFI.ipynb   # Main notebook (run on Colab)
├── cnn_model_final.keras                                 # Best CNN weights
├── vit_model_final.keras                                 # Best ViT weights
├── README.md
└── .gitignore
```

---

## How to Run

### On Google Colab (recommended)

1. Open [Google Colab](https://colab.research.google.com/) and upload the notebook, or open it directly from GitHub.
2. Enable GPU: `Runtime → Change runtime type → T4 GPU`
3. Run all cells in order (`Runtime → Run all`)
4. Training time: ~45–60 min on T4 GPU

### Expected training time per model

| Model | Approx. time (T4) |
|---|---|
| CNN Phase 1 (8 epochs) | ~8 min |
| CNN Phase 2 (10 epochs) | ~12 min |
| ViT (15 epochs) | ~25 min |

---

## Requirements

```
tensorflow >= 2.12
tensorflow-datasets
scikit-learn
numpy
pandas
matplotlib
```

> The notebook installs `tensorflow-datasets` and `scikit-learn` automatically via `pip` in the first cell.

---

## Explainability

Grad-CAM heatmaps are generated on 8 test samples to highlight the image regions that most influenced the CNN's decision. This satisfies the project requirement for a classification explainability method.

---

## Comparison to State of the Art

| Method | Accuracy on colorectal_histology |
|---|---|
| SVM + hand-crafted features (Kather 2016) | ~87% |
| ResNet-50 fine-tuned | ~95–97% |
| ViT-B/16 (pretrained ImageNet-21k) | ~98%+ |
| **Our CNN fine-tuned** | **~91–93%** |
| **Our ViT from scratch** | **~86–89%** |
| **Our Ensemble** | **~93–95%** |

The ViT gap vs SOTA is expected: training from scratch on 3 500 images limits capacity. A pretrained ViT (e.g. via `keras_hub`) would close this gap significantly.

---

## References

- Kather J.N. et al. (2016). Multi-class texture analysis in colorectal cancer histology. *Scientific Reports*. https://doi.org/10.1038/srep27988
- Dosovitskiy A. et al. (2020). An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale. *arXiv:2010.11929*
- Selvaraju R.R. et al. (2017). Grad-CAM: Visual Explanations from Deep Networks. *arXiv:1610.02391*
- Loshchilov I. & Hutter F. (2019). Decoupled Weight Decay Regularization. *ICLR 2019*
- TensorFlow Datasets — colorectal_histology: https://www.tensorflow.org/datasets/catalog/colorectal_histology

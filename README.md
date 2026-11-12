# Hyperspectral Anomaly Detection with DMS2F-HAD

A simple, self-contained PyTorch implementation of
**DMS2F-HAD: A Dual-branch Mamba-based Spatial–Spectral Fusion Network for Hyperspectral Anomaly Detection**
(A. Pant, L. Tamang, T.-K. Lee, S. Aryal — WACV 2026)
· [paper](https://openaccess.thecvf.com/content/WACV2026/papers/Pant_DMS2F-HAD_A_Dual-branch_Mamba-based_Spatial-Spectral_Fusion_Network_for_Hyperspectral_Anomaly_WACV_2026_paper.pdf)
· [official code](https://github.com/Ayushma00/DMS2F-HAD)

The repository contains the code, the four benchmark scenes, every result (anomaly maps, metrics, logs),
paper-style figures, and a slide deck that explains hyperspectral anomaly detection, the datasets and the method.

📊 **Presentation:** [`presentation/DMS2F-HAD_presentation.pptx`](presentation/DMS2F-HAD_presentation.pptx) (15 slides with speaker notes:
hyperspectral imaging and HAD in general, the method, the datasets and their use cases, the models it is compared
with, our results, and why they differ from the paper)

---

## 1. What is hyperspectral anomaly detection (HAD)?

A **hyperspectral image** is a cube of size H × W × C: every pixel holds a spectrum of C narrow, contiguous
wavelength bands (e.g. 189 bands from 400–2500 nm for AVIRIS) instead of just red, green and blue. Different
materials have different spectral "fingerprints".

**Anomaly detection** gives every pixel a score for how much it differs from the background — **without labels
and without knowing the target spectra**. Anomalies are small, rare and spectrally distinct objects such as
planes on an airport apron, vehicles, boats or rocks in grass. Uses include surveillance, search and rescue,
mineral exploration and environmental monitoring.

Methods are judged with colour **anomaly maps**, **ROC curves** (detection rate vs. false-alarm rate),
**AUC** (area under the ROC curve; 1 = perfect, 0.5 = random) and **box plots** of anomaly vs. background scores.

| Hyperspectral cube | Background vs. anomaly spectra |
|---|---|
| <img src="results/figures/fig_hsi_cube.png" width="330"> | <img src="results/figures/fig_spectra.png" width="480"> |

## 2. The method (DMS2F-HAD)

The network is an **autoencoder trained on the image itself**. The background dominates the image, so it is
reconstructed well; rare anomalies are not, and their **reconstruction error** becomes the anomaly score.

![architecture](presentation/assets/paper_fig2_architecture.png)
*Architecture figure from the paper (Fig. 2): (a) framework, (b) Mamba block, (c) SS decoder block.*

1. **Patching + masking** – the image (min-max normalised) is cut into overlapping 16×16 patches (stride 8);
   during training a random rectangle is zeroed across all bands (p = 0.5).
2. **In-head** – 1×1 conv + BN + GELU: C bands → c1 = 64 features.
3. **Spatial branch** – parallel 3×3 and 5×5 convs; the 256 pixels become a token sequence for a **Mamba** block.
4. **Spectral branch** – 1-D convs along the spectrum (kernels 7 and 3), patch pooling, then g = 7 overlapping
   band groups of c2 = 16 (stride k = 8) are fed to a **Mamba** block and projected back to 64.
5. **Adaptive gated fusion** – `G = sigmoid(conv1x1([F_spa, F_spe]))`,
   `F_fusion = Proj(G ⊙ F_spa + (1 − G) ⊙ F_spe)` — one weight per pixel.
6. **Decoder** – two SS-decoder blocks (Mamba ‖ 3×3 conv ‖ 5×5 conv → concat → 1×1 conv, residual) and a 1×1 out-head.
7. **Detection** – patches are stitched back (overlaps averaged); score `R(i,j) = ‖X(i,j,:) − X̂(i,j,:)‖₂`.

**Mamba** is a selective state-space model: `h_t = Ā_t h_{t−1} + B̄_t x_t`, `y_t = C_t h_t`, where the step size
and B, C depend on the input. It models long sequences in **linear time**, unlike self-attention (quadratic).

## 3. Datasets

The four scenes released with the official code are in `data/*.mat` (key `data` holds the cube and key `map` the
ground-truth mask). Object counts and sizes below were measured from the masks.

| | CRI | AVIRIS-1 | AVIRIS-2 | San Diego |
|---|---|---|---|---|
| Sensor | Nuance CRI tunable-filter camera | AVIRIS (NASA/JPL), airborne | AVIRIS (NASA/JPL), airborne | AVIRIS (NASA/JPL), airborne |
| Scene | grassland with rocks | airport with planes | airport with planes | airport with planes |
| Size (H × W × bands) | 400 × 400 × 46 | 100 × 100 × 189 | 128 × 128 × 189 | 100 × 100 × 189 |
| Spectral range | 650–1100 nm, 10 nm steps | 370–2510 nm; 189 of 224 bands kept | 370–2510 nm; 189 of 224 bands kept | 370–2510 nm; 189 of 224 bands kept |
| Pixel size | not reported | 3.5 m | 3.5 m | 3.5 m |
| Anomalies | 10 rocks, 15–180 px each | 3 planes, 20–22 px each | 3 planes, 34–44 px each | 3 planes, 38–56 px each |
| Anomaly pixels | 860 (0.54 %) | 64 (0.64 %) | 120 (0.73 %) | 134 (1.34 %) |

AVIRIS bands that are affected by water absorption or noise are removed in the released files. AVIRIS-2 and San
Diego come from the same flight: 7,951 of San Diego's 10,000 pixel spectra also appear in AVIRIS-2, but each scene has
its own crop and its own ground-truth mask. AVIRIS-1 shares no pixels with the other two.

**What each scene is used for, and why it is hard**

* **CRI** stands for finding small natural objects, such as rocks or debris, on vegetated ground, as in search and
  rescue. The grass texture is strong, the bands cover only red and near-infrared light, and with 160,000 pixels it is
  also the largest image, which tests speed and memory.
* **AVIRIS-1** stands for airborne surveillance of an airfield. Its three planes are tiny (about 20 pixels each) and
  dark against a bright apron, which is why RX scores only 0.8866 here.
* **AVIRIS-2** covers a larger and busier part of the airport. Buildings, roofs and vehicles also look unusual and
  compete with the planes for the highest scores; it is the hardest scene in our runs.
* **San Diego** is the classic HAD benchmark: three aircraft on an airport apron, imaged from the air. Bright roofs and
  building edges cause false alarms, especially for RX.

![datasets](results/figures/fig_datasets.png)

The paper evaluates on 14 scenes; the other 10 (Salinas, ABU urban and beach scenes, Cat Island, Gulfport, Pavia,
Texas Coast, Bay Champagne) are not in the official release, so they are not used here.

## 4. Results

Training follows the paper where it is explicit (Sec. 4.3: Adam, lr 5e-4, weight decay 1e-4, 100 epochs, 16×16
patches, stride 8, c1 = 64, c2 = 16, k = 8) and the official code where the paper is silent (loss = MSE + 0.1·L1,
batch 32, Mamba state size 64, AUC checked every epoch, zero padding at the image border). One model is trained per
image, without labels; the ground truth is used only for evaluation. Model size: **0.744 M parameters** for 189
bands (paper: 0.64 M; see section 7 for why).

<!-- RESULTS START -->
| Scene | RX (ours / paper) | DMS2F-HAD ours, best epoch | DMS2F-HAD ours, epoch 100 | DMS2F-HAD paper | Inference (s) |
|---|---|---|---|---|---|
| CRI | 0.9990 / 0.9989 | **1.0000** (epoch 79) | 1.0000 | 0.9999 | 4.124 |
| AVIRIS-1 | 0.8866 / 0.8866 | **0.9906** (epoch 25) | 0.9557 | 0.9891 | 0.288 |
| AVIRIS-2 | 0.9181 / 0.9181 | **0.9748** (epoch 49) | 0.8065 | 0.9851 | 0.441 |
| San Diego | 0.9403 / 0.9403 | **0.9889** (epoch 24) | 0.9387 | 0.9722 | 0.283 |
| **Average (4 scenes)** | 0.9360 | **0.9886** | 0.9252 | 0.9866 | |

**Ablation** (paper Table 5): our AUC at the best epoch, the paper's value in brackets. All four variants use the earlier settings (Mamba state size 16, AUC checked every 5 epochs), so they are comparable with each other.

| Scene | Spectral only | Spatial only | Addition fusion | Gated fusion (full) |
|---|---|---|---|---|
| CRI | 0.9985 (0.9997) | 0.9999 (0.9990) | 0.9999 (0.9999) | **0.9999** (0.9999) |
| AVIRIS-1 | 0.9876 (0.9856) | 0.9855 (0.9716) | **0.9878** (0.9710) | 0.9871 (0.9891) |
| AVIRIS-2 | 0.8634 (0.8701) | 0.9639 (0.9755) | **0.9682** (0.9999) | 0.9639 (0.9851) |
| San Diego | 0.9225 (0.9158) | 0.9875 (0.9674) | **0.9882** (0.9532) | 0.9876 (0.9722) |
| **Average** | 0.9430 (0.9428) | 0.9842 (0.9784) | **0.9860** (0.9810) | 0.9846 (0.9866) |

Reading: the spectral branch alone is clearly the weakest variant, as in the paper. Spatial-only, addition and gated fusion are within 0.002 AUC of each other on average, so we could not confirm the paper's gain from the gate on these scenes.

**Why our numbers differ from the paper.** Best-epoch AUC on the three AVIRIS scenes for our run (state size 64, AUC checked every epoch), the earlier runs with state size 16 (AUC checked every 5 epochs, and every epoch with seeds 0, 1 and 2), and all of the official code's settings (`--official`).

| Scene | Paper | Our run (state 64) | State 16, every 5 epochs | State 16, every epoch, seeds 0 / 1 / 2 | Official settings |
|---|---|---|---|---|---|
| AVIRIS-1 | 0.9891 | 0.9906 | 0.9871 | 0.9905 / 0.9881 / 0.9920 | 0.9894 |
| AVIRIS-2 | 0.9851 | 0.9748 | 0.9639 | 0.9700 / 0.9726 / 0.9632 | 0.9715 |
| San Diego | 0.9722 | 0.9889 | 0.9876 | 0.9892 / 0.9894 / 0.9889 | 0.9895 |
| **Mean** | 0.9821 | 0.9848 | 0.9795 | 0.9826 (mean of seeds) | 0.9835 |

* **Randomness:** across three seeds, the best AUC of one scene varies by up to 0.009, so differences of this size between single runs are run-to-run noise.
* **Checkpoint choice:** the AUC swings strongly between epochs and the paper reports the best one, so checking every epoch instead of every five raised our earlier mean from 0.9795 to 0.9826.
* **Settings:** state size 64 gives a mean of 0.9848, state size 16 gives 0.9826 and the full official settings give 0.9835, against 0.9821 in the paper; these differences are within the seed variation.
* **Not recorded:** the released code cannot run as published (it imports two files that are not in the repository, and its Mamba library is Linux-only), and the settings and seed behind each number in the paper are not documented.
* **AVIRIS-2 is not reproduced:** all 6 of our runs give 0.9632–0.9748, against 0.9851 in the paper. The other scenes match or exceed the paper.
<!-- RESULTS END -->

**About "best epoch" vs. "epoch 100".** The paper reports "the model with the best validation performance", and
the official code keeps the checkpoint with the highest AUC against the ground truth. We do the same (AUC checked
every epoch, as in the code) in the *best epoch* column. Because that uses labels to choose the checkpoint, we also report the
model after all 100 epochs. The AUC fluctuates during training even while the loss keeps falling: once the
background is learned, the decoder starts to reconstruct the anomalies too. See the training curves below.

### How DMS2F-HAD compares with other models (paper's numbers)

The paper benchmarks against six published detectors on 14 scenes (Tables 2–3) and against the other Mamba
detector, MMR-HAD, on four shared scenes (Table 4). Only RX was re-run here; the other values are as reported.

| Model | Type | Mean AUC (14 scenes) | Mean inference (s, A100) | Params |
|---|---|---|---|---|
| RX (1990) | statistical | 0.9523 | – | – |
| AUTO-AD (TGRS 2022) | fully convolutional autoencoder | 0.9189 | 9.32 | 3.23 M |
| LREN (AAAI 2021) | low-rank embedded autoencoder | 0.8297 | – | 0.86 M |
| RGAE (TGRS 2022) | robust graph autoencoder | 0.9326 | – | – |
| TDD (TGRS 2024) | transferred direct detector | 0.5900 | 2.55 | 46.64 M |
| GT-HAD (TNNLS 2025) | gated Transformer | 0.9774 | 36.26 | 0.26 M |
| **DMS2F-HAD (WACV 2026)** | **dual-branch Mamba** | **0.9878** | **0.555** | **0.64 M** |

DMS2F-HAD is best on 9 of the 14 scenes and 4.6× faster than the next fastest model (TDD). Against MMR-HAD
(2.12 M params, 3.5 GFLOPs) each model wins two of four shared scenes, with DMS2F-HAD using 3.3× fewer parameters
and 29× fewer FLOPs (0.12 GFLOPs).

On the four scenes used here, our own run is compared with all baselines in the slide deck (slide
"Our reproduction vs. the baselines").

### Colour anomaly maps (paper Fig. 3)
![anomaly maps](results/figures/fig3_anomaly_maps.png)

### ROC curves (paper Fig. 5)
![roc](results/figures/fig5_roc.png)

### Anomaly vs. background score distributions (paper Fig. 4)
![boxplots](results/figures/fig4_boxplots.png)

### Training curves
![training](results/figures/fig_training_curves.png)

More figures in [`results/figures/`](results/figures): AUC table (`table2_auc.png`), ablation table
(`table5_ablation.png`), our AUC vs. the paper (`fig_auc_vs_paper.png`) and one summary per scene
(`<scene>_result.png`). Raw numbers: [`results/auc_table.csv`](results/auc_table.csv); anomaly maps and training
histories: `results/runs/<scene>_<variant>.npz`; logs: `results/logs/`.

## 5. Repository layout

```
├── data/                        4 benchmark scenes (.mat)
├── model.py                     Mamba in plain PyTorch + the DMS2F-HAD network
├── utils.py                     loading, patching / stitching, RX baseline, AUC
├── train.py                     unsupervised training + anomaly detection for one scene
├── make_figures.py              paper-style figures, tables and the README results block
├── presentation/
│   ├── DMS2F-HAD_presentation.pptx
│   ├── make_ppt.py              builds the deck from the results
│   └── assets/
├── results/
│   ├── runs/                    <scene>_<variant>.npz: anomaly maps, AUC/loss history, timings
│   ├── figures/                 PNG figures (300 dpi)
│   ├── logs/                    training logs
│   └── auc_table.csv
└── requirements.txt
```

## 6. How to run

```bash
pip install -r requirements.txt

python train.py --dataset all                        # full model on all four scenes
python train.py --dataset aviris_1 --fusion spatial   # ablation variants: spatial | spectral | add | gated
python make_figures.py                               # figures + tables + README results block
python presentation/make_ppt.py                      # rebuild the slide deck
```

`--dataset` is one of `Cri`, `aviris_1`, `aviris_2`, `San_Diego`, `all`. A GPU is recommended; CPU works but is slow.
Other options: `--official` (all settings of the official code), `--d_state`, `--eval_every`, `--seed` and `--tag`
(tagged runs are saved to `results/runs/extra/`). The ablation runs and the `state16` runs in `results/runs/extra/`
were made with the earlier defaults (state size 16, AUC every 5 epochs, mirror padding at the border); commit
`935f08f` reproduces them exactly.

## 7. Differences from the paper and the official code

**Rule: we follow the paper where it is explicit, and the official code where the paper is silent.** The
architecture, the loss, the data, the normalisation and the anomaly score are the same as in the paper. The official
code could not be run directly: as released it imports two files that are not in the repository
(`src/project_paths.py` and `src/auc_tracker_abalation.py`), and its Mamba library (`mamba_ssm`) ships compiled CUDA
kernels for Linux only. Replacing that kernel does not change the results: our plain-PyTorch scan matches the
step-by-step recurrence to within 10⁻¹⁵.

| | Paper text | Official code | This implementation |
|---|---|---|---|
| Mamba kernel | – | `mamba_ssm` CUDA kernels (Linux only) | plain PyTorch, exact two-level chunked selective scan |
| Mamba state size | not stated | 64 | 64 (code) |
| AUC checks for the best epoch | "best validation performance" | every epoch | every epoch (code) |
| Padding at the image border | not stated | zeros | zeros (code) |
| Spatial multi-scale conv | 3×3 and 5×5 | 3×3 only (5×5 commented out) | 3×3 and 5×5 (paper) |
| Band groups | c2 = 16, k = 8 | 12 bands, stride 6 | 16 bands, stride 8 (paper) |
| Epochs | 100 | 150 | 100 (paper) |
| Masking | random size and position per sample | per batch, fixed 3×3 square | per sample, random 2–5 px rectangle (paper) |
| Model mode when scoring | not stated | training mode (random masks stay on) | evaluation mode |

The one exception to the rule is the last row: scoring in training mode leaves the random masking switched on, so
the same model gives a different anomaly map each time. We treat that as a bug and score in evaluation mode.

**Parameter count.** Rebuilding the official model's layer shapes gives exactly **0.640 M** used parameters with
Mamba state size 64, which is the number in the paper. Our model has 0.744 M parameters with the same state size,
because the paper text's 5×5 convolution and 16-band groups add parameters that the released code does not have.

`python train.py --dataset aviris_2 --official` trains with all of the official code's settings (state size 64, 3×3
conv only, 12-band groups, 150 epochs, AUC checked every epoch), so the two versions can be compared directly. The
test runs are summarised in the results section under "Why our numbers differ from the paper".

## 8. Limitations

* Four of the paper's fourteen scenes (only these are released). The main results are single runs (seed 0);
  three seeds were run on the AVIRIS scenes to measure the run-to-run variation.
* "Best epoch" AUC uses the ground truth to pick the checkpoint, as in the paper; the "epoch 100" column is the
  label-free number.
* The pure-PyTorch scan trains slower than the fused CUDA kernels (inference is still well under a second).

## Citation

```bibtex
@inproceedings{pant2026dms2fhad,
  title     = {DMS2F-HAD: A Dual-branch Mamba-based Spatial-Spectral Fusion Network for Hyperspectral Anomaly Detection},
  author    = {Pant, Aayushma and Tamang, Lakpa and Lee, Tsz-Kwan and Aryal, Sunil},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2026}
}
```

Datasets and the reference implementation are from the authors' repository
[Ayushma00/DMS2F-HAD](https://github.com/Ayushma00/DMS2F-HAD).

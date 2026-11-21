# Experiment design

**Question.** DEIM [Huang et al., CVPR 2025] argues that its Matchability-Aware Loss (MAL) is needed because Dense O2O (mosaic + mixup) creates many low-quality matches. Does that premise still hold when a COCO-pretrained DEIM-N is fine-tuned on a small dataset?

I fixed this formulation before any main run, and the analysis script (`tools/analyze_runs.py`) implements it as written here.

## 1. Factors and arms

Two independent variables, fully crossed:

- Classification loss $L \in \{\text{VFL}, \text{MAL}\}$
- Dense O2O $D \in \{\text{off}, \text{on}\}$ (on: `mosaic_prob = 0.5`, `mixup_prob = 0.5`; off: both 0)

| Arm (config in `configs/pets/`) | $L$ | $D$ | Initial weights | Seeds |
|---|---|---|---|---|
| `mal_dense` | MAL | on | DEIM-N COCO checkpoint | 0, 1, 2 |
| `vfl_dense` | VFL | on | DEIM-N COCO checkpoint | 0, 1, 2 |
| `mal_plain` | MAL | off | DEIM-N COCO checkpoint | 0 (and 1 if budget allows) |
| `vfl_plain` | VFL | off | DEIM-N COCO checkpoint | 0 (and 1 if budget allows) |
| `ref_imagenet` | MAL | on | ImageNet backbone only | 0 |

MAL and VFL runs with the same seed and the same $D$ form a pair: same data order and augmentation stream.

**Controlled variables** (identical in every arm): model (DEIM-N, HGNetv2-B0), data split, input size 640, batch size 16, learning rate (1e-4, backbone 5e-5: the COCO values scaled by 16/128), warmup (366 iterations, about 2 epochs as on COCO), EMA, 58 epochs, and the augmentation schedule below. MAL uses $\gamma = 1.5$ (DEIM default); VFL uses $\alpha = 0.75$, $\gamma = 2.0$ (D-FINE default).

| Epochs | 0–3 | 4–28 | 29–49 | 50–57 |
|---|---|---|---|---|
| Augmentation | none | Dense O2O window (only if $D$ = on) + standard | standard | none |

The reference run stops after epoch 28, the end of the Dense O2O window.

## 2. Data

Oxford-IIIT Pet, 37 breeds, head bounding boxes. Only the official trainval split is annotated (3,671 usable images), so it is split 80/10/10, stratified by breed, seed 0: 2,937 train / 367 val / 367 test (`tools/pets_to_coco.py`). Validation is used for checkpoint selection, test only for the final numbers.

## 3. Measurements

**Match quality.** For a training run $r$ and epoch $e$, let $M_{r,e}$ be the set of one-to-one Hungarian matches $(i, j)$ between predictions $\hat b_i$ and targets $b_j$ at the final decoder layer (denoising queries and auxiliary layers excluded). Over the Dense O2O window $W = \{4, \dots, 28\}$, the low-quality match fraction is

$$f_r = \frac{\sum_{e \in W} \left|\{(i,j) \in M_{r,e} : \mathrm{IoU}(\hat b_i, b_j) < 0.5\}\right|}{\sum_{e \in W} |M_{r,e}|}$$

The same window is used for the Dense O2O off arms. The mean number of targets per training image over $W$ is logged alongside.

**Accuracy.** AP, AP50, AP75 and AP$_L$ on the test split (AP$_M$ as indicative only: 72 medium boxes; there are no small boxes), from the checkpoint with the best validation AP. Also validation AP per epoch and training time per epoch.

## 4. Hypotheses and decision rule

**H1.** In this fine-tuning setting low-quality matches stay rare even under Dense O2O, so the premise behind MAL largely disappears. **H0.** The paper's account carries over: Dense O2O raises low-quality matches substantially.

**Main test (b).** For each Dense O2O run with loss $L$ and seed $s$:

$$\Delta f_{L,s} = f_{(L,\,\text{on},\,s)} - \overline{f}_{(L,\,\text{off})} \qquad \text{(b) holds for the run if } \Delta f_{L,s} < 5 \text{ percentage points}$$

where $\overline{f}_{(L,\text{off})}$ is the mean over the Dense O2O off runs with the same loss.

**Sanity check (a).** The measurement must separate strong from weak initial predictions:

$$\overline{f}_{(\text{MAL},\,\text{on})} \le \tfrac{1}{2}\, f_{\text{ref}}$$

**Decision.** H1 is supported if (b) holds for every Dense O2O run and (a) passes; rejected if (b) fails for every Dense O2O run; otherwise inconclusive.

**Secondary (estimation, not pass/fail).** For each seed pair under Dense O2O, $d_s = \mathrm{AP}_{(\text{MAL},s)} - \mathrm{AP}_{(\text{VFL},s)}$ (likewise for AP75). Report $\bar d$ with a 95% $t$ interval, $\bar d \pm t_{0.975,\,n-1}\, s_d / \sqrt{n}$. An interval excluding 0 means MAL helps (or hurts); one entirely below +0.3 means a smaller gain than on COCO; otherwise unresolved. With $n = 3$ the interval is wide, so "unresolved" is the most likely outcome.

## 5. Running

One run = one arm + one seed, launched from `notebooks/colab_run.ipynb`; outputs go to `outputs/<arm>_s<seed>/`. Then:

```
python tools/analyze_runs.py --root outputs
```

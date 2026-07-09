# Progress log and methodology

This is the living handoff document for the project. Picking this up on a fresh
machine? Read this file first, then `docs/spec.md` (the spec) and `README.md`
(the outward-facing summary). Everything needed to continue without losing
knowledge is here. Keep it updated as work proceeds.

Last updated: 2026-07-10, Stage 4 training code built and smoke-tested locally.

## Where things stand

| Stage | What | State |
|---|---|---|
| 1 | Matthew Weight loss + toy validation vs plain L2 | done |
| 2 | The paper's network, forward pass + shape/sanity tests | done |
| 3 | Wi-Pose dataloader + JHM/PAF target rendering + visual check | done |
| 4 | Train loop + PCK eval built and smoke-tested; GPU run pending | in progress |
| 5 | Synthetic CSI generator; later MM-Fi | not started |

## Environment

- Windows, Python 3.14, virtualenv at `.venv` (git-ignored).
- Install: `.venv\Scripts\python -m pip install -r requirements.txt`.
- Dependencies pinned in `requirements.txt`: torch 2.12.1 (CPU), numpy, scipy,
  matplotlib, pytest, h5py 3.16.0.
- Run tests: `.venv\Scripts\python -m pytest tests -q` (all should pass; the
  loader tests skip unless the data sample is extracted).
- CPU is fine for Stages 1 to 3. Stage 4 training goes to Colab/Kaggle (see
  the compute rule below).

## Conventions

- Commit messages: short and plain (e.g. "add readme and figures"), no
  conventional-commit prefixes.
- No em or en dashes anywhere user-facing (docs, README, comments, commits).
  Reword with commas, colons, parentheses, or separate sentences.
- Compute: keep local work to minutes-scale CPU (tests, toy runs, sample
  checks). Anything heavy (full training, sweeps, full dataset extraction) runs
  on Colab/Kaggle via GitHub, not the local machine.

## Key decisions and findings

### Stage 1: Matthew Weight loss (`piw/losses.py`)
- Weight is `w = k*|y| + b` (magnitude form). The paper's literal
  `w = k*y + b*I(y)` goes negative for negative PAF targets, which would reward
  error; the magnitude form fixes that and is identical for non-negative JHM
  targets. Constants: JHM k=1, b=1; PAF k=1, b=0.3.
- The toy experiment (`mw_vs_l2_toy.py`) was inherited unrun and had two real
  bugs found only by running it: a trailing ReLU caused dead-unit collapse
  within ~200 steps, and at the paper's blob size (sigma 1.5, 0.9% foreground)
  the toy task is unlearnable at toy scale (verified across steps, lr, encoding,
  dataset, and architecture; a raw-coordinate control proved the bottleneck is
  rendering, not decoding). The toy runs at sigma 3.0 where learning happens.
- Result: MW beats plain L2 on foreground error and joint-peak height by a
  modest margin (the 2:1 weight cap makes a large margin mathematically
  impossible), at the expected cost of noisier background. Honest, not dramatic.

### Stage 2: the network (`piw/network.py`, `piw/skeleton.py`)
- Flow: input 150x3x3 -> bilinear upsample to 150x96x96 -> residual block ->
  U-Net (one shared trunk) -> two heads at 46x82 (JHM 19ch, PAF 38ch).
  Segmentation head dropped (Wi-Pose has no masks).
- Head kernel derivation: to reach 46x82 from 96x96 with stride 2 on height and
  stride 1 on width, a valid conv needs kernel (6, 15). `(96-6)/2+1=46`,
  `(96-15)/1+1=82`.
- 8.68M parameters, ~91% in the U-Net trunk.
- Faithfulness: the paper-specified parts (input tensor, 96x96 upsample,
  residual-then-U-Net, head output shapes, stride rule) are exact. The U-Net
  interior (depth 3, base width 64, BatchNorm, bilinear-upsample decoder) is a
  standard reconstruction because the authors never released code. Tune it in
  Stage 4 if needed.
- Head channel counts come from `piw/skeleton.py` (18 joints -> JHM 19; 19 limbs
  -> PAF 38) so switching datasets is a one-line change.

### Stage 3: data (`piw/dataset.py`, `piw/targets.py`)
Findings from inspecting the real files (this is why I load one file and check
it before writing a loader; several assumptions were wrong):
- Files are **MATLAB v7.3 (HDF5)**. `scipy.io.loadmat` fails; use **h5py**.
- Variables are **`CSI`** and **`SkeletonPoints`**, not the `csi_serial` /
  `jointsVector` names in the research report.
- `CSI` is h5py shape `(3, 3, 30, 5)` = MATLAB `5x30x3x3` with axes reversed. It
  is already **amplitude** (real, non-negative), so no complex handling needed.
  To make the network input: `transpose(csi, (3,2,1,0))` -> `(5,30,3,3)` ->
  reshape `(150,3,3)`, then per-sample normalize (zero mean, unit std).
- `SkeletonPoints` is `(1, 54)` = `[vertical*18, horizontal*18, conf*18]`. The
  first block is the **vertical** (head-to-feet) axis, the second is
  **horizontal**. This was confirmed by the visual check: nose sits at a small
  block1 value and ankles at a large one, and only this ordering yields upright
  skeletons. Getting this backwards (the initial attempt) produced skeletons
  lying on their side. Source frame is portrait **480 x 640** (W x H).
- ~11.3% of joints have confidence < 0.2. These are masked out of the loss
  (per-joint for JHM, per-limb for PAF if either endpoint is low-confidence).
- Split matches the spec exactly: 132,847 Train / 33,753 Test, 12 actions,
  evenly distributed, 1405 clips total.
- Target rendering: JHM Gaussian sigma 1.5 at 46x82 plus a background channel
  (1 - max over joints); PAF is a thin unit-vector band along each valid limb.
  Verified by `stage3_data_check.py` -> `figs/stage3_targets.png`.

## Data location and handling

- Full dataset: `Wi-Pose.rar` (1.45 GB, 166,600 .mat files) lives in a local
  Downloads folder, NOT in the repo. It is on Google Drive / Baidu (see
  `docs/research_notes.md`). Never commit it.
- The repo's `data/` directory is git-ignored. For local development a sample
  was extracted: the first frame of every clip (1405 files) into
  `data/Wi-Pose_sample/Wi-Pose/{Train,Test}/`. Enough to build and visually
  verify the loader; not for training.
- `tar` on Windows 11 (build 26200+) can read RAR via libarchive:
  `tar -tf Wi-Pose.rar` to list, `tar -xf Wi-Pose.rar -C <dir> -T <listfile>`
  to extract specific members. Used this to pull the sample without unpacking
  all 166k files.
- For Stage 4, extract the full dataset on Colab/Kaggle (where training runs),
  point `WiPoseDataset` at the `{Train,Test}` root, and pull results back.

## Datasets and references: what we actually need

The research report (`docs/research_notes.md`) surveys both a software-only path
(ours) and a hardware-capture path (not ours). We need very little of it:
- **Wi-Pose**: primary dataset, in hand and verified. Its 5x30x3x3 CSI is the
  closest public match to the paper's 150x3x3 input, so the network fits with no
  adaptation. This is why we use it over the report's default (MM-Fi).
- **MM-Fi**: optional second dataset, only to measure cross-environment
  generalization later. Different geometry (114 subcarriers, 3 pairs, 17
  joints), would need head/input changes.
- Not needed: the labeling pipeline (Detectron2 / OpenPose / MMPose / PifPaf,
  because Wi-Pose ships labels), CSIKit (Wi-Pose is already processed), all CSI
  hardware (we do not capture our own data), and the other datasets
  (Person-in-WiFi-3D unless we go 3D, PerceptAlign, Wi-Mose, the HAR sets).
- DT-Pose (`cseeyangchen/DT-Pose`) has a Wi-Pose loader worth a glance as a
  cross-check, but is not a dependency.

### Stage 4: training and evaluation (in progress)

Built and smoke-tested locally:
- `piw/losses.py` gained `masked_mw_loss` (MW-weighted L2 that ignores
  masked-out channels) and `pose_loss` (JHM + PAF combined, head weights 1).
- `piw/train.py`: Adam(lr 1e-3, betas 0.9/0.999), batch 32, 20 epochs, lr halved
  at epochs 5/10/15. Checkpoints per epoch to `checkpoints/` (git-ignored).
- `piw/eval.py`: decode joints from JHM by per-channel argmax, then PCK@0.2
  normalized by the GT bbox diagonal, overall and by group (head / arms / legs).
  Note COCO-18 has no separate foot keypoints, so ankles sit in the leg group.
- `stage4_smoke.py`: end-to-end check on the sample. Loss drops cleanly
  (0.25 -> 0.008 in 90 steps) and evaluation runs; PCK is near chance (0.035),
  which is expected after 90 steps and only confirms the plumbing.

Honest observation from the smoke test: the loss falls fast partly because the
easy win is fitting the background channel and empty joint channels. Real joint
localization needs the full dataset and many epochs; watch that PCK actually
climbs during the real run, not just that loss drops. CPU is ~6 s/batch, so the
full run (about 83k batches) is GPU-only, confirming the plan.

### Review pass (2026-07-10)

Line-by-line review of the whole repo. Fixes applied:
- PCK is now measured in source image pixel space. The 46x82 grid squashes the
  portrait frame anisotropically (one grid step is ~5.9 px in x, ~13.9 px in
  y), so grid-space distances undercounted vertical error ~2.4x and the metric
  was not the paper's convention. `pck_correct` converts grid coords back to
  pixels before distances and the bbox diagonal.
- Samples with fewer than 2 confident joints (no bounding box) are now excluded
  from PCK scoring instead of silently counted as all misses.
- The background JHM channel is now computed from ALL joints including
  low-confidence ones, so it never claims "confidently empty" right where a
  masked joint probably is; the joint's own channel stays masked. Rationale in
  `piw/targets.py`.
- Training checkpoints now carry model + optimizer + scheduler + epoch, and
  `piw/train.py --resume checkpoints/epochNN.pt` continues an interrupted run
  (Colab sessions disconnect; a 20-epoch run must survive that). `piw/eval.py`
  accepts both the new checkpoint format and plain state_dicts.
- Training seeds torch and numpy (`--seed`, default 0) for reproducibility.
- The 46x82 grid size lives once in `piw/constants.py` (was duplicated in
  network.py and targets.py).
- `evaluate()` restores the model's train/eval mode instead of leaving it in
  eval.
- New tests: pixel-space anisotropy of PCK, unscoreable-sample exclusion,
  loader contract test (skips when the data sample is absent).
- Wording: losses.py docstring no longer overstates the foreground weighting.

Deferred by choice: dataset packing, pin_memory/non_blocking, sub-pixel peak
decoding, and periodic validation go into the Colab notebook work (that is
where they matter); a configurable input-channel count waits until MM-Fi.

### Colab notebook ready (2026-07-10)

Everything for the GPU run is built and dry-run locally; only the actual
20-epoch run remains. What was added:
- `piw/pack.py`: packs the 166,600 .mat files once into contiguous arrays
  (csi.npy (N,150,3,3) float32 unnormalized, skel.npy (N,54) float32,
  files.txt). Prints keypoint maxima per split as a check on the 480x640
  source-frame estimate (sample maxima: 608 vertical, 464 horizontal, so the
  estimate holds).
- `PackedWiPose` in `piw/dataset.py`: loads the packed arrays (in RAM by
  default, mmap optional) and produces items identical to `WiPoseDataset`;
  a round-trip test enforces that. Shared helpers `reshape_csi`,
  `normalize_csi`, `make_item` now back both loaders.
- `train()` and `evaluate()` accept a Dataset instance or a root path.
  `train()` gained `on_epoch_end(model, epoch)` (used for periodic validation
  from the notebook), pin_memory and non_blocking transfers.
- `decode_keypoints` now does quarter-cell sub-pixel refinement toward the
  larger neighbor (one grid cell is 6 to 14 source px, so this matters).
- `stage4_colab.ipynb`: the run itself. Order: GPU check, clone + pull, mount
  Drive (checkpoints go there so disconnects are survivable), gdown the
  dataset to the VM disk, bsdtar-extract, pack, sanity cell (asserts the
  132,847 / 33,753 split and renders one target + skeleton), train 20 epochs
  with per-epoch checkpoint + validation PCK on ~1,700 held-out frames, final
  full-test PCK (overall, per group, per joint), loss + PCK curves, artifacts
  copied to Drive. Resume after a disconnect: rerun cells, set
  resume=f"{CKPT}/epochNN.pt" in the training cell.
- Do NOT pip install requirements.txt on Colab (it pins the CPU torch build);
  Colab's preinstalled stack has everything the notebook needs.

Remaining for Stage 4: run the notebook on a GPU runtime, bring back
results_test.json, val_log.json, training_curves.png, and epoch20.pt, then
write up the numbers (README + this log). Watch for the paper's ordering
(head worst) and expect a large drop on any unseen condition (known result,
not a bug).

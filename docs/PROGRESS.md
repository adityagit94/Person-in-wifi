# Progress log and methodology

This is the living handoff document for the project. If you are picking this up
in a fresh session (any machine, any account), read this file first, then
`CLAUDE.md` (the spec) and `README.md` (the outward-facing summary). Everything
needed to continue without losing knowledge is here. Keep it updated as work
proceeds.

Last updated: 2026-07-10, after Stage 3.

## Where things stand

| Stage | What | State |
|---|---|---|
| 1 | Matthew Weight loss + toy validation vs plain L2 | done |
| 2 | The paper's network, forward pass + shape/sanity tests | done |
| 3 | Wi-Pose dataloader + JHM/PAF target rendering + visual check | done |
| 4 | Full training on the official split, report PCK@0.2 | next (cloud GPU) |
| 5 | Synthetic CSI generator; later MM-Fi | not started |

## Environment

- Windows, Python 3.14, virtualenv at `.venv` (git-ignored).
- Install: `.venv\Scripts\python -m pip install -r requirements.txt`.
- Dependencies pinned in `requirements.txt`: torch 2.12.1 (CPU), numpy, scipy,
  matplotlib, pytest, h5py 3.16.0.
- Run tests: `.venv\Scripts\python -m pytest tests -q` (17 tests, all passing).
- CPU is fine for Stages 1 to 3. Stage 4 training goes to Colab/Kaggle (see
  the compute rule below).

## Working rules (agreed with the user, keep to these)

- Commit messages: short and plain, human-sounding (e.g. "add readme and
  figures"), no conventional-commit prefixes, no verbose AI-style summaries.
- Never add `Co-Authored-By` or any AI attribution to commits. Author is the
  user alone. (History was rewritten once to remove a Claude co-author; do not
  reintroduce it.)
- No em or en dashes anywhere user-facing (docs, README, comments, commits).
  Reword with commas, colons, parentheses, or separate sentences.
- Compute: keep local work to minutes-scale CPU (tests, toy runs, sample
  checks). Anything heavy (full training, sweeps, full dataset extraction) goes
  to Colab/Kaggle via GitHub, not the user's machine. Ask before any long local
  run.
- Prefer working inline over spawning subagents/workflows (token budget).

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
Findings from inspecting the real files (this is why CLAUDE.md insists on
loading one file before writing a loader; several assumptions were wrong):
- Files are **MATLAB v7.3 (HDF5)**. `scipy.io.loadmat` fails; use **h5py**.
  CLAUDE.md's suggestion to use scipy is wrong for this dataset.
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

- Full dataset: `Wi-Pose.rar` (1.45 GB, 166,600 .mat files) lives in the user's
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

## What comes next (Stage 4)

1. On Colab/Kaggle: extract the full `Wi-Pose.rar`, point `WiPoseDataset` at the
   Train/Test root.
2. Training loop: Adam(lr 1e-3, betas 0.9/0.999), batch 32, 20 epochs, optional
   lr halving at epochs 5/10/15. Loss = MW-weighted L2 on JHM + PAF, multiplied
   by the per-channel validity masks the dataset already returns.
3. Before the real run, recompute source W,H from the full dataset maxima (the
   480x640 estimate is from the sample) if you want exact scaling.
4. Evaluate PCK@0.2 (error normalized by GT bbox diagonal) overall and per joint
   group (torso/arms, legs, head, feet). Watch for the paper's ordering (head
   and feet worst) as a sanity signal, and expect a large drop on any unseen
   condition (known result, not a bug).

# Person-in-WiFi Simulation (software-only reproduction)

This is the fixed project spec. For current status and decisions, see
`docs/PROGRESS.md`, which is the running log kept up to date as work proceeds.

## Goal
Reproduce the pipeline of "Person-in-WiFi" (ICCV 2019, arXiv:1904.00276) without WiFi hardware.
Sequence: (1) Matthew Weight loss + toy validation, (2) the paper's network, (3) train on the
public Wi-Pose dataset (real CSI, 2D pose), (4) later: a synthetic CSI generator (physics
simulation), then the MM-Fi dataset for rigor.

Context: the original authors never released code or data (IRB restrictions), so this is a
re-implementation from the paper's spec, trained on substitute data. Matching the paper's exact
numbers (mIoU 0.65, mPCK@0.2 = 78.75) is NOT expected and NOT the goal.

## Network (paper Figure 6)
- Input: 150x3x3 tensor. 150 = 5 time samples x 30 subcarriers flattened into channels;
  3x3 = transmit x receive antenna pairs.
- Bilinear upsample the input to 150x96x96 -> residual conv block -> U-Net (one shared trunk).
- Output heads downsampled to c x 46 x 82 using stride 2 on height and stride 1 on width.
- Heads used here: JHM (joint heatmaps) and PAF (part affinity fields).
  The paper's segmentation head is DROPPED: Wi-Pose ships no images or masks.
- Training config from the paper: Adam (betas 0.9, 0.999), lr 1e-3, batch size 32, 20 epochs.
  Optional: halve lr at epochs 5/10/15 (used by the sibling paper WiSPPN, arXiv:1904.00277).

## Matthew Weight (MW) loss
- Weighted L2: loss = mean( w * (pred - target)^2 )
- Weight: w = k * |y| + b
- IMPORTANT gotcha: the paper writes w = k*y + b*I(y) with I = +1 if y >= 0 else -1. Taken
  literally this goes NEGATIVE for negative PAF targets (y = -0.8, b = 0.3 gives w = -1.1),
  which would reward increasing error. Use the magnitude form above; for JHMs (y >= 0 always)
  the two forms are identical.
- Constants: JHM k=1, b=1. PAF k=1, b=0.3. Head loss weights: lambda_JHM = lambda_PAF = 1.

## Dataset: Wi-Pose (github.com/NjtechCVLab/Wi-PoseDataset)
- 166,600 .mat packets; official split 132,847 train / 33,753 test; 12 volunteers, 12 actions.
- CSI shape per packet: 5x30x3x3 (5 packets, 30 subcarriers, 3x3 antenna pairs).
  Annotations: 18 2D skeleton keypoints from AlphaPose (OpenPose COCO-18 style ordering).
- Reshape CSI 5x30x3x3 -> 150x3x3 to match the paper's input. Use amplitude only (CSI phase
  on commodity hardware is unreliable). Normalize per sample.
- FIRST TASK before any dataloader code: load ONE .mat and print variable names and shapes.
  Build the dataloader against what is actually in the file, not against assumptions. (Note:
  the files are MATLAB v7.3 / HDF5, so they need h5py, not scipy.io.loadmat.)
- Fallback if the Wi-Pose download is impractical: MM-Fi (github.com/ybhbingo/MMFi_dataset,
  Google Drive hosted, NeurIPS 2023). Note MM-Fi differs: 1 Tx x 3 Rx antennas, 114
  subcarriers, 17 joints, so the input tensor and head shapes need adapting.

## Target rendering (from keypoints)
- JHM: one channel per joint + 1 background channel. Gaussian with sigma ~1.5 px at 46x82
  resolution. Background channel = 1 - max over joint channels (OpenPose convention).
- PAF: 2 channels (x, y of a unit vector) per limb, painted within ~1 px of the limb segment,
  0 elsewhere. Use the standard OpenPose COCO limb list.
- Scale keypoint pixel coordinates to the 46x82 grid. If the source image resolution is not
  documented, estimate W and H from max coordinate values across the dataset.
- Mask joints with annotation confidence < 0.2 out of the loss (the labels come from a teacher
  network and inherit its failures).

## Evaluation
- PCK@0.2 with error normalized by the ground-truth bounding-box diagonal (box computed from
  GT keypoints), matching the paper's convention.
- Also report per-joint-group results (torso/arms, legs, head, feet). The paper found head and
  feet worst; seeing the same ordering is a good sanity signal.

## Milestones
1. MW vs plain L2 toy experiment: plot target | L2 prediction | MW prediction on synthetic
   sparse heatmaps; MW should show measurably stronger joint peaks at equal training steps.
2. Network forward pass test: random 150x3x3 in, correct head shapes out, parameter count.
3. Wi-Pose dataloader + target rendering, with a visual sanity check (rendered heatmaps and
   PAFs overlaid on the skeleton for a few samples).
4. Full training run on the official split; report PCK@0.2 on test.
5. Later: synthetic CSI generator (multipath point-scatterer model: static room paths + one
   echo per body joint, amplitude of the summed channel) feeding the same network.

## Environment
- Python venv; torch, numpy, scipy, matplotlib, h5py.
- CPU is fine for milestones 1-3. Milestone 4 (20 epochs over 132k samples) wants a GPU:
  push this repo to GitHub and train on Colab/Kaggle or an institute machine, then pull
  results back.

## Honest expectations
- In-domain (same rooms/people) results should be reasonable. Cross-environment
  generalization collapses even in the original paper (mIoU 0.65 -> 0.12 in untrained rooms),
  so a large drop on unseen conditions is the known result, not a bug in this code.

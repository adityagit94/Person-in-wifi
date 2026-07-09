# Reproducing "Person-in-WiFi" (ICCV 2019): A Complete Technical Research Report

## TL;DR
- **The original Person-in-WiFi code was never released (IRB restrictions); only data-collection tools are public.** You cannot download the authors' network or dataset, so a faithful reproduction means re-implementing the U-Net multi-task network from the paper (which is fully specified) and training it on a *different* public CSI+pose dataset — MM-Fi is the best fit — because collecting your own synchronized CSI+camera data is a multi-month hardware project.
- **The single biggest obstacle is data, not modeling.** The 3×3 MIMO / 30-subcarrier input the paper needs comes from an Intel 5300 NIC that is now discontinued; in 2026 the sane modern path is an Intel AX210 with PicoScenes or FeitCSI. An ESP32 (ESPectre) is single-antenna (1×1) and physically cannot produce the 9 antenna-pair views the method depends on.
- **For a strong undergraduate, the recommended plan is: (1) re-implement the paper's network in PyTorch and train on MM-Fi WiFi-CSI (pose only — no masks), validating the Matthew-Weight loss idea; then (2) if hardware is available, target the modern successor Person-in-WiFi 3D (CVPR 2024), whose code, dataset, and Transformer pipeline are fully released.** DensePose-from-WiFi has no official code and the viral "wifi-densepose" GitHub repos are AI-generated facades to avoid.

---

## Key Findings

1. **The paper is fully specified and reproducible on the modeling side.** Every architectural and training detail (input tensor, U-Net, three heads, Matthew Weight loss, optimizer, epochs, metrics) is in the arXiv PDF (1904.00276) and CVF open-access version. There are no published errata; the main "critique" is the paper's own honesty about cross-environment failure (mIoU collapses from 0.65 to 0.12 in untrained rooms).
2. **Official code status: NOT released.** The repo `geekfeiw/wifiperson` (MIT license; the profile listing shows 176★/38 forks, while the issue-page header still reads 181★/39 forks) contains only `datacollectioncode` and `dataprocessing` folders plus figures — the network training code is withheld. The README states verbatim: *"The paper is under review, due to IRB issues, we have not made code publicly. Still, we release data collection tools in this repo."* No dataset is hosted there.
3. **No faithful third-party reimplementation of the exact 2019 network exists.** GitHub search surfaces the author's related repos (WiSPPN, CSI-Net) and many *unrelated or fraudulent* "wifi pose/densepose" repos. The most useful genuine successor code is `aiotgroup/Person-in-WiFi-3D-repo` (Apache-2.0).
4. **MM-Fi is the practical training dataset** (NeurIPS 2023): 40 subjects, 27 actions, 4 environments, 320.76k synchronized frames, WiFi CSI (114 subcarriers, 3 Rx antennas) + 17-joint 2D/3D pose, CC BY-NC 4.0, direct Google Drive/Baidu download. **Caveat: MM-Fi provides no segmentation masks**, so the paper's segmentation head can only be reproduced by generating your own masks or omitting it.
5. **CSI hardware in 2026: AX210 + PicoScenes/FeitCSI is the modern standard.** The Intel 5300 + Halperin Linux 802.11n CSI Tool that the paper used is legacy (needs old kernels, discontinued NIC). Only NICs with ≥3 Rx antennas (IWL5300, QCA9300, or an antenna array) give the 3×3 MIMO the paper requires.
6. **ESPectre is an excellent learning tool but architecturally cannot reproduce the paper.** A single ESP32 is 1×1 (one antenna pair, 64 subcarriers), giving 1 view, not 9; multiple ESP32s cannot be phase-synchronized. It is valuable for cheaply learning the CSI pipeline and reusable signal-processing patterns.

---

## Details

### 1. The Paper Itself — full method

**Sensing setup and physical intuition.** Two off-the-shelf 802.11n WiFi devices are used: one transmitter set (T) and one receiver set (R), each with **3 antennas lined up like a household router**. This gives **3×3 = 9 antenna (propagation) pairs**. CSI is recorded at **30 EM frequencies (subcarriers)** in a 20 MHz channel centered at **2.4 GHz** (wavelength ≈ 12.5 cm). The three receiver antennas are **spaced uniformly within one wavelength (12.5 cm)** to maximize the difference of CSI captured across antennas. CSI is sampled at **100 Hz**; the RGB annotation camera runs at **20 FPS**, so each video frame corresponds to **5 sequential CSI samples**. This yields 3×3×30×5 = **1350 "equations"** per frame to reconstruct ~10⁴ image unknowns — the core "ill-posed 1D→2D" problem the paper addresses by (a) adding more views/frequencies and (b) constraining the mapping with multiple spatial supervision targets.

**Input tensor.** The network input is a **150×3×3 tensor**: the 150 = 5 time samples × 30 subcarriers is flattened onto the first (channel) dimension, and the 3×3 spatial dims are the antenna pairs. The paper does **2D convolution along the 3×3 antenna-pair dimension**, arguing the 9 views encode a coarse 2D spatial layout.

**Network (Figure 6 in the paper).**
- Input 150×3×3 is **upsampled to 150×96×96**.
- Fed to a **residual convolution block**, then a **U-Net** (Ronneberger et al. 2015).
- U-Net outputs are **downsampled to c×46×82** using kernels with **stride 2 on height and stride 1 on width** to match the ground-truth resolution.
- The downsampling receptive field (size 140) exceeds the 96×96 upsampled map, so every feature "sees" all 9 antenna views.
- One shared U-Net serves SM and JHMs (they found this as good as two separate U-Nets).

**Three output heads (all c×46×82):**
- **Segmentation Mask (SM): 1×46×82** — binary person mask, from Mask R-CNN annotations.
- **Joint Heat Maps (JHMs): 26×46×82** — 25 OpenPose Body-25 joints + 1 background.
- **Part Affinity Fields (PAFs): 52×46×82** — 26 limbs × 2 (x,y) components.

**Multi-task loss.** L = λ₁·L_SM + λ₂·L_JHM + λ₃·L_PAF, with **λ₁ = 0.1, λ₂ = λ₃ = 1**. L_SM uses **Binary Cross-Entropy**. L_JHM and L_PAF use a **weighted L2**:
- L_JHM^(i,j,c) = w_(i,j,c) · ‖ŷ − y‖₂²

**Matthew Weight (MW).** The paper's key trick. Plain L2 fails because **98% of JHM pixels are background** (CDF of the 26×46×82 = 98,072-element tensor shows <2% belong to joints), so L2 averages error over background and washes out joints. The Matthew Weight (named after the "rich get richer" Matthew effect) is:
- **w_(i,j,c) = k·y_(i,j,c) + b·𝕀(y_(i,j,c))**, where 𝕀 outputs +1 if y ≥ 0 else −1.
- **JHM: k=1, b=1. PAF: k=1, b=0.3.** MW puts higher weight on larger (joint) values, forcing attention onto the sparse foreground. Figure 7 shows MW dramatically improves pose vs. plain L2 without needing cascaded stages (OpenPose) or stacked hourglass networks.

**Training.** PyTorch, **Adam (β₁=0.9, β₂=0.999)**, **batch size 32**, **initial LR 0.001**, **20 epochs**. Multi-person joint association uses the OpenPose Python API on the predicted JHMs/PAFs, outputting p×25×3 (x, y, confidence).

**Data.** >10⁵ frames (154,627 total: 99,366 one-person, plus 2–5 person groups), **16 indoor scenes** (6 lab office + 10 classroom), 8 volunteers. First experiment: **first 80% of each group for training (123,631), last 20% for test (30,996)** — same identities/environments, different poses.

**Evaluation metrics and reported results.**
- **Segmentation:** mIoU and mAP over AP@50–AP@95 (COCO-style). Reported **mIoU 0.65, mAP 0.38** (AP@50 = 0.91 down to AP@90–95 ≈ 0), i.e., torsos/legs segment well, subtle masks poorly.
- **Pose:** modified **PCK** (normalized by person bbox diagonal, bbox from aligning OpenPose joints with Mask R-CNN box). Reported **mPCK@0.20 = 78.75** (Table 3 lists 78.75; the same table lists Person-in-WiFi mIoU 0.66). Gaps vs. teachers: Mask R-CNN mIoU 0.83, OpenPose mPCK@0.20 89.48.
- Joint groups: Torso&Arms and Legs score highest; Head and Feet lowest (small parts, diffraction at 12.5 cm wavelength).

**Failure modes (paper's own analysis):** (1) lack of spatial resolution / small limbs missed due to diffraction; (2) rare poses; (3) incomplete single-camera annotations (camera 70° FOV vs. WiFi 360°).

**GAN-based environment invariance.** Preliminary. Step 1: pre-train a binary **environment discriminator D** that takes a pair of CSI tensors and outputs 1 if same environment, 0 otherwise. Step 2: fix D, train a **U-Net generator G** so that generator outputs (GCSI) fool D into "same environment," while GCSI simultaneously feeds the Person-in-WiFi net (both trained jointly). On 14 train / 2 test scenes this improved untrained-environment **mIoU 0.12→0.24** and **mPCK@0.20 19.34→31.06** — still far below in-domain, confirming cross-environment generalization is unsolved.

**Errata / critiques / reproductions.** No formal errata. There is no known faithful public reproduction of the exact network. Community discussion (e.g., issues on `geekfeiw/wifiperson`) mostly asks for code/data that were never released. The honest limitation flagged by the authors and later literature is severe environment-dependence.

### 2. Official Code and Reproductions

- **`github.com/geekfeiw/wifiperson`** (MIT; profile shows 176★/38 forks) — Official repo. Contains `datacollectioncode`, `dataprocessing`, `figs`, README. Languages: Python 71.7%, MATLAB 21.7%, C 5.3%. So you get CSI collection scripts (built around the Intel 5300 CSI Tool) and annotation-processing scaffolding — **not** the network or trained weights.
- The README documents the annotation pipeline they used verbatim: *"we use a Mask R-CNN implementation, detectorch to prepare human mask, and OpenPose python-api to prepare human pose... with help of OpenPose developers, Gines and Raaj."* (i.e., `ignacio-rocco/detectorch` for masks; **OpenPose Python API (Body-25)** for JHMs/PAFs).
- Related author repos (Fei Wang, `geekfeiw`): **WiSPPN** (`Can WiFi Estimate Person Pose?` test example, ~128★), **CSI-Net** (PyTorch, with data + pre-trained models), **ARIL**, **Multi-Scale-1D-ResNet**. WiSPPN is the closest in spirit (WiFi→pose adjacency-matrix) and does ship some code/weights.
- **Third-party reimplementations of the exact 2019 heads/MW loss:** none found that are trustworthy. Many "wifi-densepose" repos (ruvnet/RuView and its many forks: yangsuzhou, ljq, davidakpele, euaziel) are **AI-generated and non-functional** — an independent audit fork (`deletexiumu/wifi-densepose`) documents that the core pipeline "returns random/hardcoded data, neural network models have no trained weights, and the claimed performance metrics are fabricated." **Avoid these.**
- The genuinely useful modern codebase is **`aiotgroup/Person-in-WiFi-3D-repo`** (Apache-2.0), built on OpenMMLab (MMCV/MMDetection), implementing the CVPR 2024 successor.

### 3. Datasets (critical — train without collecting your own)

**MM-Fi (recommended) — `github.com/ybhbingo/MMFi_dataset`, arXiv:2305.10345 (NeurIPS 2023 D&B).**
- **Scale:** 40 subjects (11F/29M), **27 actions** (14 daily + 13 rehab), **4 environments**, **1080 sequences**, **320.76k synchronized single-person frames** (each sequence 297 frames @ 10 Hz).
- **Modalities:** RGB, depth, LiDAR point cloud, mmWave radar point cloud, **WiFi CSI**.
- **WiFi hardware/format:** two TP-Link N750 APs + **Atheros CSI Tool**, **5 GHz / 40 MHz**, **114 subcarriers**, **1 Tx antenna / 3 Rx antennas** (3 pairs), hardware up to 1000 Hz. Stored as MATLAB `.mat`, per-sequence shape **297×3×114**; augmented per-100 ms sample **3×114×10**. Contains some `−inf` values (handled by interpolation in the official toolbox).
- **Annotations:** 2D keypoints (17 joints, HRNet-w48) and **3D keypoints (17 joints**, triangulated, PCKh@0.5 = 95.66%), 3D position, algorithm-generated 3D dense pose (unvalidated), 27 action labels + temporal segments. **No segmentation masks.**
- **Download:** open Google Drive + Baidu Netdisk links (no request form for core data; raw RGB requires an application form). **License: CC BY-NC 4.0** (research/non-commercial). IRB-approved (NTU IRB-2022-1067). DOI 10.5281/zenodo.7983467. (Exact `MMFi_Dataset.zip` byte size is not published.)
- **Mapping to this task:** WiFi CSI (3×114×samples) → 17-joint pose. To use the paper's architecture, treat CSI as the input tensor (adapting 30→114 subcarriers, 3×3→3×1 or 3×3 depending on Tx config) and the 17 joints as JHM/PAF targets. **Because there are no masks, drop the SM head or generate masks via Detectron2 on the (application-gated) RGB frames.**

**Person-in-WiFi 3D dataset (aiotgroup, CVPR 2024) — best modern target.**
- **97K+ frames**, **7 volunteers**, 1–3 persons, 8 daily actions, three distinct locations, 4 m × 3.5 m area. **14 joints with 3D coordinates.** Officially partitioned into **89,946 WiFi samples for training and 7,824 for testing**. Hardware: 4× ThinkPad X201 laptops (**1 Tx + 3 Rx**, Intel 5300 NICs).
- **Download:** processed WiFi+pose (BaiduNetdisk, code `k50e`); raw (BaiduNetdisk, code `xjtu`); sdp8 mirror. Directory `data/wifipose/{train_data,test_data}/{csi,keypoint,*_list.txt}`.
- Code Apache-2.0; dataset CC BY-NC 4.0. Reported 3D joint errors (verbatim, Yan et al. CVPR 2024): *"91.7mm (1-person), 108.1mm (2-person), and 125.3mm (3-person), comparable to cameras and millimeter-wave radars."*

**WiPose / Wi-Pose (2D pose).**
- Publicly downloadable version: **`github.com/NjtechCVLab/Wi-PoseDataset`** — 166,600 packets (.mat), 12 actions, 12 volunteers. Camera + 5 GHz router + WiFi host; pose from AlphaPose. CSI shape **5×30×3×3** (5 packets, 30 subcarriers, 3×3 antennas). 18 skeleton keypoints (2D). Frames: 132,847 train / 33,753 test. **This CSI layout (5×30×3×3) is the closest public match to the Person-in-WiFi input tensor** — a good sanity-check dataset. (Note: the original 3D "WiPose" from Jiang et al., MobiCom 2020, Buffalo, is listed as not publicly accessible; naming is ambiguous.)
- The `cseeyangchen/DT-Pose` repo aggregates loaders for MM-Fi, WiPose, and Person-in-WiFi-3D datasets — useful reference code.

**Wi-Mose (3D moving pose).** Xie et al., "3D Moving Human Pose Estimation Using Commodity WiFi" (IEEE SPL 2021, arXiv:2012.14066). Fuses CSI amplitude+phase into a "CSI image." P-MPJPE 29.7 mm (LoS) / 37.8 mm (NLoS). No widely-mirrored public dataset download found.

**PerceptAlign (newest, cross-domain 3D pose) — `github.com/Trymore-lab/PerceptAlign` (MIT).** 21 participants, multiple indoor scenes, 17 action classes; Intel 5300 (1 Tx, 3 Rx, **57 subcarriers**, `.mat`) + RealSense D435; 3D SMPL-X keypoints + camera calibration + tx/rx geometry. Open **Hugging Face** downloads (Scene1–5 + keypoints). arXiv:2601.12252 — recent/forthcoming (2026), treat as bleeding-edge.

**HAR-only datasets (activity labels, NOT pose/segmentation — useful only for pipeline practice):**
- **UT-HAR** (`ermongroup/Wifi_Activity_Recognition`) — Intel 5300, 30 subcarriers, 3×3, 7 activities, continuous (no golden segmentation). Sample 3×30×250.
- **Widar 3.0** — Intel 5300, 30 subcarriers, 258K gesture instances, 75 domains; includes BVP (body-coordinate velocity profile) features. Gesture recognition.
- **SignFi** — Intel 5300, 5 GHz/20 MHz, 276 sign gestures, 5520/2760/7500 clips. Sign-language.
- **NTU-Fi** (part of SenseFi, `xyanchen/WiFi-CSI-Sensing-Benchmark`) — Atheros tool, TP-Link N750, **114 subcarriers**, 6 activities (HAR) + 14 gaits (Human-ID). Higher-resolution CSI.
- These have **no pose or segmentation labels** — they cannot train Person-in-WiFi outputs, but are good for validating your CSI parsing/preprocessing.

### 4. CSI Extraction Hardware and Tools (2024–2026 state of the art)

| Tool | NIC / chip | Subcarriers | MIMO | Bands/BW | 2026 status / notes |
|---|---|---|---|---|---|
| **Linux 802.11n CSI Tool** (Halperin, `dhalperi/linux-80211n-csitool`) | Intel IWL5300 | **30 subcarrier groups** (of 56/114) | up to 3×3 | 2.4/5 GHz, 20/40 MHz | Legacy: needs old kernels (orig. 2.6.36; `spanev` fork works to ~4.15/Ubuntu 18.04). **NIC discontinued — hard to source in 2026 (second-hand only).** This is exactly what the paper used. |
| **Atheros CSI Tool** (`xieyaxiongfly`) | Atheros AR9300/QCA9558 etc. (ath9k) | **56 (20 MHz) / 114 (40 MHz)** | up to 3×3 | 2.4/5 GHz | Open-source, no firmware mod; runs on Ubuntu/OpenWRT. Source builds increasingly awkward on modern kernels. Used by MM-Fi and NTU-Fi. |
| **Nexmon CSI** (`seemoo-lab/nexmon_csi`) | Broadcom/Cypress (RPi 3B+/4 bcm43455c0, some phones) | **64 (20) / 128 (40) / 256 (80 MHz)** incl. guard/null | 1×1 on RPi (up to 4×4 on bcm4366c0) | 2.4/5 GHz, up to 80 MHz | Actively maintained; RPi 5 / recent kernels supported via `Makefile.rpi` (discussion #395). Cheap. But RPi is single-antenna → not 3×3. |
| **ESP32-CSI-Tool** (`StevenMHernandez/ESP32-CSI-Tool`) | ESP32 | **64** (20 MHz) | **1×1 only** | 2.4 GHz, 20/40 MHz | ~€5–10. Active (2 ESP32s: STA-TX + AP-RX) or passive mode. ESP-IDF v4.3. ~45–100 pkt/s. CSV output. Single antenna pair. |
| **PicoScenes** (`ps.zpj.io`, Jiang et al.) | IWL5300, QCA9300, **AX200/AX210**, USRP/HackRF | up to **1992+ per stream** (up to 8192 for AX210 6 GHz) | up to 4×4; **up to 27 NICs concurrently** | 2.4/5/6 GHz, up to 160 MHz, 802.11a/g/n/ac/ax(/be on SDR) | Most versatile. Ubuntu 20.04, needs SSE4.2/AVX2. Note EULA collects some usage stats. MATLAB/Python toolboxes. |
| **FeitCSI** (`feitcsi.kuskosoft.com`, `KuskoSoft/FeitCSI`) | Intel **AX200/AX210** (+likely newer) | all 802.11ax subcarriers (up to 1992 @160 MHz) | per NIC (2×2 typical for AX200/210) | 2.4/5/6 GHz, 20/40/80/160 MHz | **First fully open-source** AX200/210 CSI+injection tool; GUI/CLI/UDP; live USB; any Linux arch; no data collection. Younger project than PicoScenes. |

- **CSIKit** (`Gi-z/CSIKit`, MIT) — Python library parsing **Atheros, Intel (IWL5300/AX200/AX210), Nexmon, ESP32, FeitCSI, PicoScenes** formats into numpy; returns `(frames, subcarriers, rx, tx)` matrices; includes Hampel/low-pass filters. **Use this as your unified CSI loader.**
- **Getting 3×3 MIMO (the paper's requirement):** only IWL5300, QCA9300 (3-antenna), or a **multi-NIC AX210 array via PicoScenes** provide ≥3 Rx antennas with 3 Tx. AX200/AX210 are typically **2×2**; to get the full 9 pairs you'd run **PicoScenes with multiple synchronized AX210 NICs** (its distinctive multi-NIC feature) or accept a reduced antenna configuration and adapt the input tensor. MM-Fi itself uses **1 Tx × 3 Rx** (3 pairs), not 9.
- **Antenna spacing:** ~**12.5 cm = one wavelength at 2.4 GHz** (paper spaced 3 Rx antennas uniformly within one wavelength to maximize inter-antenna CSI diversity). At 5 GHz the wavelength is ~6 cm, so spacing shrinks accordingly.
- **Recommended modern setup:** **Intel AX210 (~$20–30 each) + PicoScenes or FeitCSI**, on a wired Linux box, with a second AX210/AP as transmitter. Total hardware for a 2-NIC rig is roughly **$60–150** (2× AX210 + adapters/antennas + a spare mini-PC), far cheaper than sourcing multiple IWL5300s. For 3×3 fidelity, budget for 3 receiver NICs and PicoScenes multi-NIC.

### 5. The Labeling Pipeline (generating ground-truth targets from a synchronized camera)

The paper's supervision is **auto-generated from a synchronized RGB camera** — you never hand-label poses. Pipeline:
- **Segmentation masks (SM):** run **Mask R-CNN**. The paper used `detectorch`; the modern maintained choice is **Detectron2** (`facebookresearch/detectron2`) with a COCO-pretrained Mask R-CNN (e.g., `mask_rcnn_R_50_FPN_3x`). Detectron2 installs via pip/conda with matching PyTorch+CUDA; run inference per frame, take the person mask, resize to 1×46×82.
- **Joint Heat Maps + PAFs (JHM/PAF):** the paper used **OpenPose Body-25** (25 joints + background → 26 JHM channels; 26 limbs × 2 → 52 PAF channels). OpenPose is notoriously hard to build (Caffe deps, CUDA) and is **non-commercial-license only** — the CMU license states: *"The non-exclusive commercial license requires a non-refundable USD 25,000 annual royalty. The non-exclusive commercial license cannot be used in the field of Sports."* The modern maintained alternative is **MMPose** (`open-mmlab/mmpose`, Apache-2.0), which supports top-down and bottom-up estimators and can emit heatmaps and PAF-like association fields; or **PifPaf** (`openpifpaf`), which natively produces PIF+PAF fields. For a faithful reproduction you want a **bottom-up, heatmap+PAF** model (Body-25/COCO-25 or COCO-17); if you switch to 17 joints (COCO), adjust JHM to 18 channels and PAF to 2×#limbs accordingly.
- **Time synchronization:** CSI at **100 Hz** and video at **20 FPS** → **5 CSI samples per frame**. Sync by timestamps (the paper aligns on recorded timestamps). Practical approach: log a common monotonic clock on the capture host, timestamp each CSI packet and each video frame, then nearest-neighbor/interpolate CSI to each frame; group the 5 CSI samples preceding each frame into the temporal dimension. Clock drift between the CSI host and camera is the usual culprit for misalignment — use one host if possible.
- **CSI parsing:** **CSIKit** to read whatever format your NIC produces, then amplitude (and optionally phase, sanitized) → the input tensor. Apply **Hampel filtering** (outlier removal) and low-pass smoothing before feeding the network (CSIKit ships these).

### 6. ESP32 / ESPectre Relevance

- **ESPectre** (`francescopace/espectre`, GPLv3; the francescopace profile now shows ~8.7k★/662 forks) — Wi-Fi CSI **motion/presence detection** on ESP32 with **ESPHome/Home Assistant** integration. Features: **NBVI (Normalized Band Variance Index)** auto-selection of **12 non-consecutive subcarriers** (claims **F1 > 96%** with zero config), **Hampel filtering** (MAD-based outlier removal), **gain lock** (AGC/FFT stabilization), two detectors — **MVS** (Movement Variance Segmentation, statistical) and **ML** (a compact MLP over **9 turbulence features**: mean, std, max, min, iqr, skewness, autocorr, mad, waveform_length → net **9→32→16→1**), and **ping/DNS traffic generation** (~100 pkt/s) to force CSI packets. Runs on ESP32/C3/S3/C6.
- **Companion:** **`francescopace/micropython-esp32-csi`** — a MicroPython fork exposing ESP32 CSI to Python (the foundation for "Micro-ESPectre," the R&D/prototyping platform).
- **Fundamental limitation for this paper:** an ESP32 has **one antenna (1×1)** and yields **64 subcarriers from a single antenna pair = 1 view**. Person-in-WiFi needs **9 antenna-pair views** (3 Tx × 3 Rx). You **cannot** reproduce fine-grained segmentation/pose on a single ESP32. **Synchronizing multiple ESP32s does not help**, because independent radios have no shared phase reference (each has its own oscillator/CFO), so you cannot form a coherent MIMO array — you'd get uncalibrated, phase-incoherent single-view streams.
- **What ESPectre IS good for here:** (1) **learning the entire CSI pipeline cheaply** (~€10) end-to-end; (2) **reusable signal-processing patterns** — Hampel filtering, NBVI subcarrier selection, gain-lock, traffic generation — that transfer directly to your MM-Fi/AX210 preprocessing; (3) a **coarse motion/presence stepping stone** before attempting fine-grained perception. Treat it as pedagogy and preprocessing inspiration, not a sensing front-end for the paper.

### 7. Modern Successors and Better Targets

- **Person-in-WiFi 3D (CVPR 2024, aiotgroup)** — the direct successor by the same lead author (Fei Wang). **More WiFi devices** (1 Tx + 3 Rx laptops, Intel 5300) + a **Transformer / DETR-style bottom-up "Wi-Fi Pose Transformer" (PETR-based)** for **end-to-end multi-person 3D pose**. Modules: Wi-Fi encoder → pose decoder (query-based) → refine decoder; set-based **Hungarian loss**. Notably, the authors report (verbatim) that a naïve "3D-ified" Person-in-WiFi — *"we represented multi-person poses with 3D keypoint heatmaps ∈14×64×64×64 and 3D part affinity fields ∈42×64×64×64... However, this deep network failed to converge"* — which motivated the Transformer redesign. **Code (Apache-2.0), dataset, and pretrained pipeline are released.** This is the **best modern target**: real code, real data, active method.
- **DensePose From WiFi (CMU, arXiv:2301.00250, Jiaqi Geng, Dong Huang, Fernando De la Torre, submitted Dec 31 2022; 13 pp/10 figs)** — maps CSI to **dense body-surface UV maps** (SMPL) using **3 Tx × 3 Rx → 3×3 feature map**, a modality-translation network (CSI→image-like domain), then a DensePose-RCNN-style head; the abstract claims performance "comparable to image-based approaches" (AP@50 ≈ 87.2 on real data per secondary reporting). The setup "only requires 2 of these routers [e.g. TP-Link AC1750]... around 30 dollars" each. **No official code or dataset released.** The many "wifi-densepose" GitHub repos are **not** from CMU and are AI-generated facades (see §2). Good to read for method ideas; not reproducible from official artifacts.
- **RF-Pose / RF-Pose3D (MIT, Zhao et al., CVPR 2018 / SIGCOMM 2018)** — **FMCW radar** (not WiFi), T-shaped antenna arrays, 1.78 GHz bandwidth, through-wall 2D/3D pose. Context only: requires specialized SDR radar hardware, not commodity WiFi; not reproducible on a student budget.
- Other WiFi→pose with code/data: **MetaFi/MetaFi++**, **GoPose**, **WiViPose**, **DT-Pose** (`cseeyangchen/DT-Pose`, aggregates datasets), **MultiFormer**, **PerceptAlign** (§3). For pure activity recognition with strong benchmark code: **SenseFi** (`xyanchen/WiFi-CSI-Sensing-Benchmark`).
- **Recommendation:** For a third-year undergrad, **first re-implement the 2019 paper's network on MM-Fi (2D/3D pose head, MW loss)** as a tractable, well-scoped exercise with guaranteed data; **then move to Person-in-WiFi 3D** as the modern target with released code — a much better use of effort than chasing DensePose-from-WiFi (no code) or the 2019 dataset (unavailable).

### 8. Practical Reproduction Roadmap

**Stage 0 — Understand & prototype (1–2 weeks).** Read the arXiv PDF; implement the network from the spec (input 150×3×3 → upsample 150×96×96 → residual block → U-Net → three downsampled heads). Implement BCE (SM) and the **Matthew-Weight weighted-L2** (JHM k=1,b=1; PAF k=1,b=0.3). Unit-test the MW against plain L2 on a toy sparse-heatmap target to confirm it concentrates loss on foreground.

**Stage 1 — Train on public data (2–4 weeks).** Download **MM-Fi** (WiFi-CSI + 17-joint 3D pose). Adapt the input tensor to MM-Fi's 3×114×samples (or reshape to the paper's convention). Generate JHM/PAF targets from the provided keypoints (Gaussian heatmaps for joints; vector fields along limb segments for PAFs). **Drop the SM head** (MM-Fi has no masks) or, if you have RGB access, generate masks with **Detectron2**. Validate with PCK / MPJPE. Use MM-Fi's official train/test splits (random / cross-subject / cross-environment) to measure generalization honestly.

**Stage 2 — (Optional) Own hardware capture.** If a professor wants real capture: **2–3× Intel AX210 + PicoScenes** (or FeitCSI), a synchronized RGB camera, and the **Detectron2 + MMPose/PifPaf** labeling pipeline. Space Rx antennas ~one wavelength apart. Expect this to be the **dominant time sink**.

**Stage 3 — Modern target.** Reproduce **Person-in-WiFi 3D** with the released `aiotgroup` code + dataset.

**Realistic effort:** Modeling + MM-Fi training is achievable in **~4–8 weeks** for a strong PyTorch student. A full own-hardware collection akin to the paper (16 scenes, 100k+ synchronized frames, working CSI toolchain) is a **multi-month** endeavor and the main risk.

**Common pitfalls (ranked):**
1. **Getting CSI to flow reliably** — driver/firmware/kernel matching (IWL5300 legacy kernels; AX210 needs PicoScenes/FeitCSI), traffic generation to force packets, sustaining a stable packet rate. This defeats most beginners first.
2. **Cross-environment generalization failure** — the paper's own numbers (mIoU 0.65→0.12 in new rooms) show models overfit environment multipath. Budget for domain-adaptation (the GAN approach, or newer adversarial/geometry-aware methods like PerceptAlign) if deployment matters.
3. **CSI phase noise / calibration** — raw CSI phase has CFO/SFO/STO offsets and random phase; use amplitude primarily, and sanitize phase (linear-fit removal, conjugate multiplication across antennas) if used. ESP32 phase is especially unreliable.
4. **Time synchronization** — CSI (100 Hz) vs. video (20 FPS) drift; prefer a single capture host and timestamp everything on one clock.
5. **Label quality** — auto-labels from a single camera are incomplete under occlusion/360° (paper's failure mode 3); consider multi-camera if collecting your own.

---

## Recommendations

**Do this, in order:**
1. **Re-implement the network and Matthew-Weight loss now** from the arXiv spec — no data or hardware needed to build and unit-test it. Confirm MW beats plain L2 on synthetic sparse heatmaps. *(Threshold to proceed: MW visibly concentrates gradient on foreground joints.)*
2. **Train on MM-Fi WiFi-CSI → pose** (17 joints). MM-Fi ships a PyTorch dataloader (CSIKit only needed if converting formats). Report PCK/MPJPE on the random, cross-subject, and cross-environment splits. **Omit the segmentation head** (no masks in MM-Fi) unless you generate masks from RGB with Detectron2. *(Threshold: sane in-domain PCK; expect a large cross-environment drop — that's the known result, not a bug.)*
3. **Only if the professor wants real capture, buy 2–3× Intel AX210 and use PicoScenes or FeitCSI** plus a Detectron2 + MMPose/PifPaf labeling pipeline. Do **not** attempt to source Intel 5300s in 2026, and do **not** rely on a single ESP32 for fine-grained perception.
4. **For a higher-impact modern project, pivot to Person-in-WiFi 3D (CVPR 2024)** — released code (Apache-2.0), released dataset (89,946 train / 7,824 test WiFi samples, 14 joints), Transformer method. This is the better long-term target.
5. **Use ESPectre + ESP32-CSI-Tool as a cheap teaching rig** to learn the CSI pipeline, Hampel filtering, and subcarrier selection — not as the paper's sensor.

**Benchmarks that change the plan:** If cross-environment PCK is acceptable on MM-Fi with the base network, invest in the GAN/domain-adaptation module. If CSI capture on AX210 proves flaky after ~2 weeks, fall back to training purely on public datasets. If multi-person is a goal, skip the 2019 architecture and go straight to Person-in-WiFi 3D.

---

## Caveats

- **The 2019 dataset and network weights are unavailable** (IRB); any "reproduction" trains a re-implementation on a *different* dataset, so exact number-matching to mIoU 0.65 / mPCK@0.20 78.75 is not expected.
- **MM-Fi has no segmentation masks and uses 114 subcarriers / 1 Tx×3 Rx**, not the paper's 30 subcarriers / 3×3 — so it is a close but not identical substitute; the segmentation head and the exact 9-view input require adaptation.
- **Avoid the viral "wifi-densepose"/RuView GitHub repos** — independently audited as non-functional AI-generated code with fabricated metrics.
- **DensePose-From-WiFi has no official code or data.**
- **PerceptAlign and some cited surveys are dated 2026 and may be forthcoming/not-yet-peer-reviewed** — treat as bleeding-edge.
- **ESP32 single-antenna physics is a hard limit**, not a tuning problem; no amount of firmware work makes one ESP32 produce 9 coherent MIMO views.
- Some performance figures (ESPectre F1>96%, various repo benchmarks) and star/fork counts are **self-reported and change over time** and are environment-dependent; validate independently before relying on them.
- The exact **MMFi_Dataset.zip file size** is not published in official sources.

---

## Quick reference — names, one-liners, links

- **Person-in-WiFi (ICCV 2019)** — the paper. arXiv:1904.00276 ; CVF open access.
- **geekfeiw/wifiperson** — official repo, data-collection tools only. github.com/geekfeiw/wifiperson
- **geekfeiw/WiSPPN** — "Can WiFi Estimate Person Pose?" test code. github.com/geekfeiw/WiSPPN
- **MM-Fi** — 5-modality (incl. WiFi CSI) + 3D pose dataset. github.com/ybhbingo/MMFi_dataset ; arXiv:2305.10345
- **Person-in-WiFi 3D (CVPR 2024)** — multi-person 3D pose, Transformer, released code+data. aiotgroup.github.io/Person-in-WiFi-3D ; github.com/aiotgroup/Person-in-WiFi-3D-repo
- **DensePose From WiFi (CMU 2023)** — dense pose, no code. arXiv:2301.00250
- **WiPose (public)** — 2D pose, CSI 5×30×3×3. github.com/NjtechCVLab/Wi-PoseDataset
- **PerceptAlign (2026)** — cross-domain 3D pose, HF downloads. github.com/Trymore-lab/PerceptAlign
- **SenseFi / NTU-Fi** — HAR benchmark + datasets. github.com/xyanchen/WiFi-CSI-Sensing-Benchmark
- **Linux 802.11n CSI Tool** — IWL5300, 30 subcarriers. github.com/dhalperi/linux-80211n-csitool ; spanev fork for newer kernels
- **Atheros CSI Tool** — ath9k, 56/114 subcarriers.
- **Nexmon CSI** — Broadcom/RPi, 64/128/256 subcarriers. github.com/seemoo-lab/nexmon_csi
- **ESP32-CSI-Tool** — ESP32, 64 subcarriers, 1×1. github.com/StevenMHernandez/ESP32-CSI-Tool
- **PicoScenes** — AX210/AX200/QCA9300/IWL5300/SDR, multi-NIC. ps.zpj.io
- **FeitCSI** — open-source AX200/AX210 CSI+injection. feitcsi.kuskosoft.com ; github.com/KuskoSoft/FeitCSI
- **CSIKit** — Python CSI parser for all formats. github.com/Gi-z/CSIKit
- **ESPectre** — ESP32 CSI motion detection (learning tool). github.com/francescopace/espectre
- **micropython-esp32-csi** — MicroPython fork exposing ESP32 CSI. github.com/francescopace/micropython-esp32-csi
- **Detectron2** — Mask R-CNN for mask labels. github.com/facebookresearch/detectron2
- **MMPose** — maintained pose estimator (OpenPose alternative). github.com/open-mmlab/mmpose
- **OpenPose** — Body-25 (non-commercial license). github.com/CMU-Perceptual-Computing-Lab/openpose
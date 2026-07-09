"""COCO-18 skeleton definition (OpenPose body ordering).

Wi-Pose ships 18 2D keypoints in OpenPose COCO-18 style ordering. The exact
joint semantics get confirmed against a real .mat file in stage 3 (the rule is
to build against what is actually in the file, not assumptions). For now these
definitions fix the two numbers the network heads need: how many joints and
how many limbs.

Head channel counts follow the OpenPose convention:
  JHM = num_joints + 1   (one heatmap per joint plus a background channel)
  PAF = num_limbs * 2    (an (x, y) unit-vector field per limb)
"""

# 18 keypoints, index -> name.
JOINT_NAMES = [
    "nose",       # 0
    "neck",       # 1
    "r_shoulder", # 2
    "r_elbow",    # 3
    "r_wrist",    # 4
    "l_shoulder", # 5
    "l_elbow",    # 6
    "l_wrist",    # 7
    "r_hip",      # 8
    "r_knee",     # 9
    "r_ankle",    # 10
    "l_hip",      # 11
    "l_knee",     # 12
    "l_ankle",    # 13
    "r_eye",      # 14
    "l_eye",      # 15
    "r_ear",      # 16
    "l_ear",      # 17
]

# Standard OpenPose COCO limb list (19 connections). The last two
# (shoulder-ear) are the redundant links OpenPose includes to stabilize PAF
# association. Each pair indexes into JOINT_NAMES.
LIMBS = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10),
    (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17),
    (2, 16), (5, 17),
]

NUM_JOINTS = len(JOINT_NAMES)   # 18
NUM_LIMBS = len(LIMBS)          # 19

JHM_CHANNELS = NUM_JOINTS + 1   # 19
PAF_CHANNELS = NUM_LIMBS * 2    # 38

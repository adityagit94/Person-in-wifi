"""Stage 2 deliverable: random 150x3x3 in, correct head shapes out, param count.

Run: python stage2_network_check.py
"""

import torch

from piw.network import IN_CHANNELS, PersonInWiFi, count_parameters


def main():
    torch.manual_seed(0)
    model = PersonInWiFi().eval()

    x = torch.randn(2, IN_CHANNELS, 3, 3)
    up = PersonInWiFi.upsample_input(x)
    with torch.no_grad():
        jhm, paf = model(x)

    print("Person-in-WiFi network, forward pass check")
    print("-" * 46)
    print(f"input                 {tuple(x.shape)}")
    print(f"after bilinear upsample{'':1}{tuple(up.shape)}")
    print(f"JHM head (joints+bg)  {tuple(jhm.shape)}")
    print(f"PAF head (limbs x 2)  {tuple(paf.shape)}")
    print("-" * 46)

    total = count_parameters(model)
    parts = {
        "residual block": count_parameters(model.res_block),
        "U-Net trunk": count_parameters(model.unet),
        "JHM head": count_parameters(model.jhm_head),
        "PAF head": count_parameters(model.paf_head),
    }
    for name, n in parts.items():
        print(f"{name:>16}: {n:>12,}")
    print(f"{'TOTAL':>16}: {total:>12,}")


if __name__ == "__main__":
    main()

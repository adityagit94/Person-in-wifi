"""Training loop for the Person-in-WiFi pose network.

Adam (betas 0.9/0.999), lr 1e-3, batch 32, 20 epochs, lr halved at epochs
5/10/15 (the sibling paper WiSPPN's schedule). Loss is the masked
Matthew-Weighted L2 over JHM and PAF. CPU is fine for a smoke test on the
sample; the full 20-epoch run over 132k frames wants a GPU (see docs/PROGRESS.md).
"""

import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

from piw.dataset import WiPoseDataset
from piw.losses import pose_loss
from piw.network import PersonInWiFi


def train(root, epochs=20, batch=32, lr=1e-3, device="cpu", workers=0,
          max_steps=None, ckpt="checkpoints", log_every=20):
    ds = WiPoseDataset(root, "Train")
    dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=workers,
                    drop_last=True)
    model = PersonInWiFi().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[5, 10, 15],
                                                 gamma=0.5)
    os.makedirs(ckpt, exist_ok=True)
    print(f"train frames: {len(ds)}  batches/epoch: {len(dl)}  device: {device}",
          flush=True)

    model.train()
    history = []
    step = 0
    stop = False
    for epoch in range(1, epochs + 1):
        running, n, t0 = 0.0, 0, time.time()
        for bd in dl:
            csi = bd["csi"].to(device)
            jhm, paf = bd["jhm"].to(device), bd["paf"].to(device)
            jm, pm = bd["jhm_mask"].to(device), bd["paf_mask"].to(device)

            jhm_p, paf_p = model(csi)
            loss, lj, lp = pose_loss(jhm_p, jhm, jm, paf_p, paf, pm)
            opt.zero_grad()
            loss.backward()
            opt.step()

            running += loss.item()
            n += 1
            step += 1
            if step % log_every == 0:
                print(f"  epoch {epoch} step {step}  loss {loss.item():.4f}  "
                      f"(jhm {lj.item():.4f}  paf {lp.item():.4f})", flush=True)
            if max_steps and step >= max_steps:
                stop = True
                break

        sched.step()
        avg = running / max(n, 1)
        history.append(avg)
        print(f"[epoch {epoch}] avg loss {avg:.4f}  ({time.time() - t0:.1f}s)",
              flush=True)
        torch.save(model.state_dict(), os.path.join(ckpt, f"epoch{epoch:02d}.pt"))
        if stop:
            break
    return model, history


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",
                    default=os.path.join("data", "Wi-Pose_sample", "Wi-Pose"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--ckpt", default="checkpoints")
    a = ap.parse_args()
    train(a.root, a.epochs, a.batch, a.lr, a.device, a.workers, a.max_steps,
          a.ckpt)

"""Training loop for the Person-in-WiFi pose network.

Adam (betas 0.9/0.999), lr 1e-3, batch 32, 20 epochs, lr halved at epochs
5/10/15 (the sibling paper WiSPPN's schedule). Loss is the masked
Matthew-Weighted L2 over JHM and PAF. CPU is fine for a smoke test on the
sample; the full 20-epoch run over 132k frames wants a GPU (see docs/PROGRESS.md).
"""

import argparse
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from piw.dataset import WiPoseDataset
from piw.losses import pose_loss
from piw.network import PersonInWiFi


def train(data, epochs=20, batch=32, lr=1e-3, device="cpu", workers=0,
          max_steps=None, ckpt="checkpoints", log_every=20, seed=0,
          resume=None, on_epoch_end=None):
    """Train the network.

    ``data`` is either a Dataset instance (PackedWiPose for real runs) or a
    root path, in which case WiPoseDataset(data, "Train") is used.
    ``on_epoch_end(model, epoch)``, if given, runs after each epoch's
    checkpoint is saved (e.g. periodic validation from the notebook).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    ds = data if isinstance(data, Dataset) else WiPoseDataset(data, "Train")
    dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=workers,
                    drop_last=True, pin_memory=(device != "cpu"),
                    generator=torch.Generator().manual_seed(seed))
    model = PersonInWiFi().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[5, 10, 15],
                                                 gamma=0.5)
    os.makedirs(ckpt, exist_ok=True)

    start_epoch, history = 1, []
    if resume:
        state = torch.load(resume, map_location=device)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        start_epoch = state["epoch"] + 1
        history = list(state.get("history", []))
        print(f"resumed from {resume} (finished epoch {state['epoch']})",
              flush=True)
    print(f"train frames: {len(ds)}  batches/epoch: {len(dl)}  "
          f"device: {device}  seed: {seed}", flush=True)

    model.train()
    step = 0
    stop = False
    for epoch in range(start_epoch, epochs + 1):
        running, n, t0 = 0.0, 0, time.time()
        for bd in dl:
            csi = bd["csi"].to(device, non_blocking=True)
            jhm = bd["jhm"].to(device, non_blocking=True)
            paf = bd["paf"].to(device, non_blocking=True)
            jm = bd["jhm_mask"].to(device, non_blocking=True)
            pm = bd["paf_mask"].to(device, non_blocking=True)

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
        # full training state, so an interrupted run resumes with --resume
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": epoch,
                    "history": history},
                   os.path.join(ckpt, f"epoch{epoch:02d}.pt"))
        if on_epoch_end is not None:
            on_epoch_end(model, epoch)
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
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None,
                    help="checkpoint to continue from (epochNN.pt)")
    a = ap.parse_args()
    train(a.root, a.epochs, a.batch, a.lr, a.device, a.workers, a.max_steps,
          a.ckpt, seed=a.seed, resume=a.resume)

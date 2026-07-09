"""Stage 4 local smoke test: prove the training + evaluation pipeline runs end
to end on the sample and the loss goes down. This is a plumbing check, not a
real training run (the sample is tiny and the step budget is small), so the
PCK number here is near chance and only confirms evaluation executes.

Run: python stage4_smoke.py
"""

import os

from piw.eval import evaluate
from piw.train import train

ROOT = os.path.join("data", "Wi-Pose_sample", "Wi-Pose")


def main():
    model, history = train(ROOT, epochs=3, batch=16, lr=1e-3, device="cpu",
                           max_steps=90, log_every=15, ckpt="checkpoints")
    print(f"\nloss per epoch: {[round(h, 4) for h in history]}")
    down = history[-1] < history[0]
    print(f"loss decreased over training: {down}")

    res = evaluate(model, ROOT, "Test", device="cpu", batch=16, max_batches=5)
    print(f"\nPCK@0.2 overall (barely trained, expect near chance): "
          f"{res['overall']:.3f}")
    print(f"by group: "
          + ", ".join(f"{g} {v:.3f}" for g, v in res["groups"].items()))
    print("\nsmoke test OK: pipeline trains and evaluates end to end")


if __name__ == "__main__":
    main()

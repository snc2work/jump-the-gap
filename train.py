"""Train the world model using only image sequences and action IDs."""
import argparse
from pathlib import Path

import numpy as np
import torch

from model import WorldModel, loss_for_frames


def train(data: Path, output: Path, steps: int, batch_size: int, seed: int, layers: int) -> None:
    torch.manual_seed(seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    with np.load(data / "train.npz") as dataset:
        frames = torch.from_numpy(dataset["frames"])
        actions = torch.from_numpy(dataset["actions"]).long()
    model = WorldModel(layers).to(device)
    # Keep minibatches identical when comparing model depths.
    batches = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    print(f"Training {sum(p.numel() for p in model.parameters()):,} parameters on {device}.", flush=True)

    for step in range(1, steps + 1):
        rows = torch.randint(len(frames), (batch_size,), generator=batches)
        images = frames[rows].to(device)
        commands = actions[rows].to(device)
        weights, logits = model(images[:, :-1], commands)
        loss = loss_for_frames(weights, logits, images[:, 1:])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == steps:
            print(f"Step {step}/{steps}: next-frame NLL {loss.item():.3f}", flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("data/transformer.pt"))
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layers", type=int, choices=(2, 3), default=2)
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        parser.error("--steps and --batch-size must be positive")
    train(args.data, args.output, args.steps, args.batch_size, args.seed, args.layers)

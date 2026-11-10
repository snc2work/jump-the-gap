"""Export the held-out model rollouts as a standalone browser replay."""
import argparse
import base64
import json
from pathlib import Path

import numpy as np
import torch

from evaluate import landing_outcome, paired_cases
from model import WorldModel


SAMPLES = 12


def encode(frames: np.ndarray) -> str:
    return base64.b64encode(np.packbits(frames).tobytes()).decode("ascii")


@torch.inference_mode()
def export(data: Path, checkpoint: Path, output: Path, seed: int) -> None:
    model = WorldModel(layers=3)
    model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu"))
    model.eval()
    torch.manual_seed(seed)
    with np.load(data / "test.npz") as dataset:
        frames = torch.from_numpy(dataset["frames"]).bool()
        actions = torch.from_numpy(dataset["actions"]).long()
    selected, _ = paired_cases(data, frames, actions, seed)

    cases = []
    for number, (rows, jump, terrain, course) in enumerate(selected, start=1):
        row = int(rows[0])
        prefix = frames[row:row + 1, :jump + 1].expand(SAMPLES, -1, -1, -1)
        plans = actions[row:row + 1].expand(SAMPLES, -1)
        futures = model.rollout(prefix, plans)[:, jump + 1:].numpy()
        labels = [landing_outcome(future, terrain, course) for future in futures]
        both = "stopped" in labels and "fell_after_landing" in labels
        pair = [labels.index("stopped"), labels.index("fell_after_landing")] if both else [0, 1]
        cases.append({
            "number": number,
            "gap": int(course.gap),
            "width": int(course.width),
            "run": jump,
            "actions": actions[row, jump:].tolist(),
            "current": encode(frames[row, jump].numpy()),
            "samples": [{"frames": encode(future), "outcome": label}
                        for future, label in zip(futures, labels, strict=True)],
            "pair": pair,
            "both": both,
        })

    cases.sort(key=lambda case: not case["both"])
    template = Path(__file__).with_name("preview_template.html").read_text(encoding="utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template.replace("__ROLLOUT_DATA__", json.dumps(cases)), encoding="utf-8")
    print(f"Exported {len(cases)} cases; both outcomes visible in {sum(case['both'] for case in cases)}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint", type=Path, default=Path("data/compare-3layers.pt"))
    parser.add_argument("--output", type=Path, default=Path("data/preview.html"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    export(args.data, args.checkpoint, args.output, args.seed)

"""Export the three-layer model and held-out observations for browser inference."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from evaluate import paired_cases
from export_preview import encode
from model import CONTEXT, WorldModel


class BrowserStep(nn.Module):
    """Read the newest token of a causally masked, right-padded context."""
    def __init__(self, model: WorldModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, frames: Tensor, actions: Tensor, index: Tensor) -> tuple[Tensor, Tensor]:
        weights, logits = self.model(frames, actions)
        return weights.index_select(1, index)[0, 0], logits.index_select(1, index)[0, 0]


def export(data: Path, checkpoint: Path, output: Path) -> None:
    model = WorldModel(layers=3)
    model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu"))
    model.eval()
    step = BrowserStep(model)
    torch.backends.mha.set_fastpath_enabled(False)
    with np.load(data / "test.npz") as dataset:
        frames = torch.from_numpy(dataset["frames"])
        actions = torch.from_numpy(dataset["actions"]).long()
    selected, _ = paired_cases(data, frames.bool(), actions, seed=42)

    seeds = []
    seen = set()
    for rows, jump, _, course in selected:
        geometry = (int(course.gap), int(course.width))
        if geometry in seen:
            continue
        seen.add(geometry)
        row = int(rows[0])
        seeds.append({
            "gap": geometry[0],
            "width": geometry[1],
            "observations": encode(frames[row, :jump + 1].numpy()),
            "actions": actions[row, :jump].tolist(),
        })
    seeds.sort(key=lambda scene: abs(scene["gap"] - 6))

    row = int(selected[0][0][0])
    jump = selected[0][1]
    example = (frames[row:row + 1, :CONTEXT], actions[row:row + 1, :CONTEXT], torch.tensor([jump]))
    padded_frames = example[0].clone()
    padded_actions = example[1].clone()
    padded_frames[:, jump + 1:] = 0
    padded_actions[:, jump + 1:] = 0
    with torch.no_grad():
        expected = model(frames[row:row + 1, :jump + 1], actions[row:row + 1, :jump + 1])
        actual = step(padded_frames, padded_actions, example[2])
        for full, short in zip(actual, expected, strict=True):
            torch.testing.assert_close(full, short[0, -1], atol=1e-5, rtol=1e-5)

    output.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        step, example, output / "world.onnx", dynamo=False, opset_version=17,
        input_names=["frames", "actions", "index"], output_names=["weights", "logits"],
    )
    (output / "seeds.json").write_text(json.dumps(seeds), encoding="utf-8")
    print(f"Exported model and {len(seeds)} held-out starting states to {output}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint", type=Path, default=Path("data/compare-3layers.pt"))
    parser.add_argument("--output", type=Path, default=Path("web/public/model"))
    args = parser.parse_args()
    export(args.data, args.checkpoint, args.output)

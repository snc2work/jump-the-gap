"""Check image prediction and sampled futures on held-out courses."""
import argparse
from pathlib import Path

import numpy as np
import torch

from model import IMAGE_SIZE, WorldModel
from simulator import Action, Course, SURFACE


def landing_outcome(frames: np.ndarray, terrain: np.ndarray, course: Course) -> str:
    """Read visible contact and run-off events without requiring exact timing."""
    if not np.all(frames[:, terrain]):
        return "invalid"
    positions = []
    for frame in frames & ~terrain:
        ys, xs = frame.nonzero()
        if not len(xs):
            positions.append(None)
            continue
        if len(xs) > 2 or xs[0] != xs[-1] or ys[-1] - ys[0] != len(xs) - 1:
            return "invalid"
        if len(xs) == 1 and ys[0] != IMAGE_SIZE - 1:
            return "invalid"
        positions.append((xs[-1], ys[-1] + 1))

    landed = False
    fell = False
    left_screen = False
    for position in positions:
        if position is None:
            left_screen = True
            continue
        if left_screen:
            return "invalid"
        x, foot = position
        landed |= course.near_edge <= x < course.far_edge and foot == SURFACE
        fell |= landed and x >= course.far_edge and foot > SURFACE
    if fell:
        return "fell_after_landing"
    last = positions[-1]
    if last is not None and positions[-3:] == [last] * 3:
        x, foot = last
        if course.near_edge <= x < course.far_edge and foot == SURFACE:
            return "stopped"
    return "other"


def paired_cases(data: Path, frames: torch.Tensor, actions: torch.Tensor, seed: int):
    with np.load(data / "test_evaluation.npz") as metadata:
        group_ids = metadata["group"]
        outcomes = metadata["outcome"]
        courses = metadata["course"]
    expected = {"stopped", "fell_after_landing"}
    paired = []
    for rows in np.split(np.arange(len(group_ids)), np.flatnonzero(np.diff(group_ids)) + 1):
        if not expected <= set(outcomes[rows]):
            continue
        row = int(rows[0])
        jump = int((actions[row] == Action.JUMP).nonzero()[0].item())
        terrain = frames[row, 0].numpy().copy()
        terrain[:SURFACE] = False
        course = Course(*courses[row])
        # A sub-frame contact can be invisible at this resolution.
        observed_outcomes = {
            landing_outcome(frames[reference, jump + 1:].numpy(), terrain, course)
            for reference in rows
        }
        if expected <= observed_outcomes:
            paired.append((rows, jump, terrain, course))
    if not paired:
        raise RuntimeError("No paired cases with visibly distinct landing outcomes.")
    rng = np.random.default_rng(seed)
    selected = [paired[index] for index in rng.choice(len(paired), size=min(24, len(paired)), replace=False)]
    return selected, outcomes


@torch.inference_mode()
def evaluate(data: Path, checkpoint: Path, seed: int, layers: int) -> None:
    model = WorldModel(layers)
    model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu"))
    model.eval()
    torch.manual_seed(seed)
    with np.load(data / "test.npz") as dataset:
        frames = torch.from_numpy(dataset["frames"]).bool()
        actions = torch.from_numpy(dataset["actions"]).long()

    exact = exact_copy = changed_frames = exact_changed = 0
    for observed, commands in zip(frames.split(16), actions.split(16), strict=True):
        weights, logits = model(observed[:, :-1], commands)
        mode = weights.argmax(-1)[..., None, None].expand(-1, -1, 1, IMAGE_SIZE**2)
        prediction = (logits.gather(-2, mode).squeeze(-2) > 0).reshape_as(observed[:, 1:])
        matches = (prediction == observed[:, 1:]).flatten(2).all(-1)
        changed = (observed[:, :-1] != observed[:, 1:]).flatten(2).any(-1)
        exact += int(matches.sum())
        exact_copy += int((~changed).sum())
        changed_frames += int(changed.sum())
        exact_changed += int((matches & changed).sum())

    total = actions.numel()
    print(f"Next frame exact: {exact / total:.1%}; copy baseline: {exact_copy / total:.1%}")
    print(f"Exact on changing frames: {exact_changed / changed_frames:.1%}")

    selected, outcomes = paired_cases(data, frames, actions, seed)
    expected = {"stopped", "fell_after_landing"}
    samples = 12
    diverse = exact_either = exact_both = visible_either = visible_both = 0
    for rows, jump, terrain, course in selected:
        row = int(rows[0])
        prefix = frames[row:row + 1, :jump + 1].expand(samples, -1, -1, -1)
        plans = actions[row:row + 1].expand(samples, -1)
        futures = model.rollout(prefix, plans)[:, jump + 1:]
        diverse += int(futures.flatten(1).unique(dim=0).size(0) > 1)
        exact_matches = (futures[:, None] == frames[rows, jump + 1:]).flatten(2).all(-1).any(0)
        matched = set(outcomes[rows[exact_matches.numpy()]])
        exact_either += int(bool(expected & matched))
        exact_both += int(expected <= matched)
        visible = {landing_outcome(future, terrain, course) for future in futures.numpy()}
        visible_either += int(bool(expected & visible))
        visible_both += int(expected <= visible)

    print(f"Distinct sampled futures: {diverse}/{len(selected)} paired cases")
    print(f"Exact simulator clips among {samples} draws, either outcome: {exact_either}/{len(selected)}")
    print(f"Exact simulator clips among {samples} draws, both outcomes: {exact_both}/{len(selected)}")
    print(f"Visible landing outcomes, either: {visible_either}/{len(selected)}; both: {visible_both}/{len(selected)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint", type=Path, default=Path("data/transformer.pt"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layers", type=int, choices=(2, 3), default=2)
    args = parser.parse_args()
    evaluate(args.data, args.checkpoint, args.seed, args.layers)

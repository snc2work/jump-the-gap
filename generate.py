"""Generate paired action-conditioned clips with disjoint course splits."""
import argparse
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np

from simulator import Action, Course, simulate


STEPS = 32
KICKS = (-0.20, -0.07, 0.07, 0.20)
BRAKE_DELAYS = (0, 2, None)


def action_plan(run_steps: int, brake_delay: int | None) -> np.ndarray:
    actions = np.full(STEPS, Action.COAST, dtype=np.uint8)
    actions[:run_steps] = Action.RUN
    actions[run_steps] = Action.JUMP
    if brake_delay is not None:
        actions[run_steps + 11 + brake_delay:] = Action.BRAKE
    return actions


def generate(output: Path, seed: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for split in ("train", "test"):
        frame_rows, action_rows, group_ids, outcome_rows, course_rows = [], [], [], [], []
        outcomes = Counter()
        mixed_stop_and_fall = 0
        for gap, width in product(range(3, 11), range(2, 7)):
            held_out = (gap + width) % 4 == 0
            if held_out != (split == "test"):
                continue
            course = Course(gap, width)
            for start_tile, run_steps, brake_delay in product(range(2, 8), range(4, 12), BRAKE_DELAYS):
                start_x = start_tile + rng.uniform(0.15, 0.85)
                actions = action_plan(run_steps, brake_delay)
                group_id = len(frame_rows) // len(KICKS)
                group_outcomes = set()
                prefix = None
                for kick in KICKS:
                    frames, outcome = simulate(course, start_x, actions, kick)
                    if prefix is None:
                        prefix = frames[:run_steps + 1]
                    elif not np.array_equal(prefix, frames[:run_steps + 1]):
                        raise RuntimeError("Paired clips must have identical observations before jumping.")
                    frame_rows.append(frames)
                    action_rows.append(actions)
                    group_ids.append(group_id)
                    outcome_rows.append(outcome)
                    course_rows.append((gap, width))
                    outcomes[outcome] += 1
                    group_outcomes.add(outcome)
                if {"stopped", "fell_after_landing"} <= group_outcomes:
                    mixed_stop_and_fall += 1

        if not mixed_stop_and_fall or outcomes["unfinished"]:
            raise RuntimeError(f"Invalid {split} split: {dict(outcomes)}")
        np.savez_compressed(
            output / f"{split}.npz",
            frames=np.asarray(frame_rows),
            actions=np.asarray(action_rows),
        )
        np.savez_compressed(
            output / f"{split}_evaluation.npz",
            group=np.asarray(group_ids),
            outcome=np.asarray(outcome_rows),
            course=np.asarray(course_rows),
        )
        print(f"{split}: {len(frame_rows)} clips, {mixed_stop_and_fall} paired stop/fall groups")
        print(dict(outcomes))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate(args.output, args.seed)

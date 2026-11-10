"""Reference simulator for generating training and evaluation video."""
from dataclasses import dataclass
from enum import IntEnum
from math import floor

import numpy as np


SIZE = 32
LEFT_EDGE = 13
SURFACE = 19
GRAVITY = 0.32


class Action(IntEnum):
    COAST = 0
    RUN = 1
    JUMP = 2
    BRAKE = 3


@dataclass(frozen=True)
class Course:
    gap: int
    width: int

    @property
    def near_edge(self) -> int:
        return LEFT_EDGE + self.gap

    @property
    def far_edge(self) -> int:
        return self.near_edge + self.width

    def supports(self, x: float) -> bool:
        return x < LEFT_EDGE or self.near_edge <= x < self.far_edge


@dataclass
class Player:
    x: float
    y: float = SURFACE
    vx: float = 0.0
    vy: float = 0.0
    grounded: bool = True
    landed: bool = False

    def step(self, course: Course, action: Action, jump_kick: float) -> None:
        if self.grounded:
            if action == Action.JUMP:
                self.vx = max(0.15, self.vx + jump_kick)
                self.vy = -1.68
                self.grounded = False
            else:
                if action == Action.RUN:
                    self.vx = min(1.35, self.vx + 0.18)
                else:
                    friction = 0.42 if action == Action.BRAKE else 0.13
                    self.vx = max(0.0, self.vx - friction)
                self.x += self.vx
                self.grounded = course.supports(self.x)
                return

        x_before, y_before = self.x, self.y
        self.x += self.vx
        self.y += self.vy + 0.5 * GRAVITY
        self.vy += GRAVITY

        if y_before <= SURFACE <= self.y and self.y > y_before:
            fraction = (SURFACE - y_before) / (self.y - y_before)
            contact_x = x_before + fraction * self.vx
            if course.supports(contact_x):
                self.landed |= contact_x >= course.near_edge
                self.y = SURFACE
                self.vy = 0.0
                self.grounded = course.supports(self.x)
                return

        if x_before < course.near_edge <= self.x:
            fraction = (course.near_edge - x_before) / self.vx
            contact_y = y_before + fraction * (self.y - y_before)
            if contact_y > SURFACE:
                self.x = course.near_edge - 0.01
                self.vx = 0.0

    def frame(self, course: Course) -> np.ndarray:
        frame = np.zeros((SIZE, SIZE), dtype=np.uint8)
        frame[SURFACE:, :LEFT_EDGE] = 1
        frame[SURFACE:, course.near_edge:course.far_edge] = 1
        x, foot = floor(self.x), floor(self.y)
        if 0 <= x < SIZE and 0 < foot <= SIZE + 1:
            frame[max(0, foot - 2):min(SIZE, foot), x] = 1
        return frame


def simulate(
    course: Course, start_x: float, actions: np.ndarray, jump_kick: float
) -> tuple[np.ndarray, str]:
    player = Player(start_x)
    frames = np.empty((len(actions) + 1, SIZE, SIZE), dtype=np.uint8)
    frames[0] = player.frame(course)
    for index, action in enumerate(actions, start=1):
        player.step(course, Action(action), jump_kick)
        frames[index] = player.frame(course)

    if player.grounded and player.x < LEFT_EDGE and player.vx == 0:
        return frames, "stopped_on_left"
    if player.landed:
        if player.grounded and player.vx == 0:
            return frames, "stopped"
        if not player.grounded:
            return frames, "fell_after_landing"
    if not player.landed and player.y > SURFACE:
        outcome = "overshot" if player.x >= course.far_edge else "fell_short"
        return frames, outcome
    return frames, "unfinished"

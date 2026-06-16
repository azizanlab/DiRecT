"""
Central obstacle definitions for the avoiding-v0 environment.

All coordinates are in real-world (unnormalized) space.
Import from here — do not duplicate obstacle geometry elsewhere.
"""

from dataclasses import dataclass


@dataclass
class CircularObstacle:
    """Circular keep-out zone defined by center and radius."""
    center: tuple  # (x, y)
    radius: float


@dataclass
class PlanarObstacle:
    """Half-plane constraint: slope * x + intercept - y >= 0 is the feasible side."""
    point_a: tuple  # (x, y)
    point_b: tuple  # (x, y)

    @property
    def slope(self):
        return (self.point_b[1] - self.point_a[1]) / (self.point_b[0] - self.point_a[0])

    @property
    def intercept(self):
        return self.point_b[1] - self.slope * self.point_b[0]

    def violated(self, x, y):
        return y > self.slope * x + self.intercept


# ---------------------------------------------------------------------------
# Obstacle sets
# ---------------------------------------------------------------------------

PILLARS = [
    CircularObstacle(center=(0.5, -0.1), radius=0.03),
    CircularObstacle(center=(0.425, 0.08), radius=0.025),
    CircularObstacle(center=(0.575, 0.08), radius=0.025),
    CircularObstacle(center=(0.35, 0.26), radius=0.025),
    CircularObstacle(center=(0.5, 0.26), radius=0.025),
    CircularObstacle(center=(0.65, 0.26), radius=0.025),
]

NOVEL_CIRCULAR = [
    CircularObstacle(center=(0.5, -0.1), radius=0.08),
    # CircularObstacle(center=(0.5, 0.16), radius=0.02),
    # CircularObstacle(center=(0.4, 0.175), radius=0.02),
    # CircularObstacle(center=(0.6, 0.175), radius=0.02),
]

NOVEL_PLANAR = [
    PlanarObstacle(point_a=(0.8, -0.3), point_b=(0.575, 0.5)),
    PlanarObstacle(point_a=(0.2, -0.3), point_b=(0.425, 0.5)),
]

CONSTRAINT_SETS = {
    "novel": {
        "circular": PILLARS + NOVEL_CIRCULAR,
        "planar": NOVEL_PLANAR,
    },
}


def count_constraints(constraint_name):
    if constraint_name == "" or constraint_name not in CONSTRAINT_SETS:
        return 0
    cs = CONSTRAINT_SETS[constraint_name]
    return len(cs["circular"]) + len(cs["planar"])


TARGET_Y = 0.35  # y coordinate of the arrival/goal region


def check_violation(observation, constraint_name):
    """Check if a single observation violates any constraint in the given set.

    Returns 1.0 if violated, 0.0 otherwise.
    Constraints are not checked once the agent reaches the target region (y >= TARGET_Y).
    """
    import numpy as np

    if constraint_name == "":
        return 0.0

    if constraint_name not in CONSTRAINT_SETS:
        raise NotImplementedError(f"Constraint '{constraint_name}' is not implemented.")

    x = observation[2]
    y = observation[3]

    if y >= TARGET_Y:
        return 0.0

    constraints = CONSTRAINT_SETS[constraint_name]

    for obs in constraints["circular"]:
        distance = np.sqrt((x - obs.center[0]) ** 2 + (y - obs.center[1]) ** 2)
        if distance <= obs.radius:
            return 1.0

    for obs in constraints["planar"]:
        if obs.violated(x, y):
            return 1.0

    return 0.0

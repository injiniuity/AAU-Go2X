"""Pure route-planning helpers used by the navigation runtime."""

import math
from collections.abc import Callable


Pose = dict


def normalize_yaw(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def with_rotated_yaw(pose: Pose, delta_yaw: float) -> Pose:
    rotated = dict(pose)
    rotated["yaw"] = normalize_yaw(float(pose.get("yaw", 0.0)) + delta_yaw)
    return rotated


def door_approach_sequence(door_name: str, find_exact: Callable[[str], Pose | None]) -> list[Pose]:
    door_name = str(door_name).strip()
    if not door_name:
        return []
    sequence = []
    waypoint = find_exact(f"{door_name}_waypoint1") or find_exact(f"{door_name}_waypoint")
    if waypoint is not None:
        sequence.append(waypoint)
    door_pose = find_exact(door_name)
    if door_pose is not None:
        sequence.append(door_pose)
    return sequence

import math
import re
from typing import Any


def normalize_name_key(text: str) -> str:
    return " ".join(str(text).strip().casefold().split())


def build_room_lookup(room_entry_config: dict[str, dict[str, Any]]) -> dict[str, str]:
    lookup = {}
    for room_id, room_config in room_entry_config.items():
        for name in [*room_config.get("doors", []), *room_config.get("members", [])]:
            lookup[normalize_name_key(name)] = room_id
    return lookup


def ascii_safe_text(text: str) -> str:
    return str(text).encode("ascii", errors="backslashreplace").decode("ascii")


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitize_for_json(val) for key, val in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def sanitize_for_tts(text: str) -> str:
    text = re.sub(r"[*_`#]", " ", text)
    text = re.sub(r"[\u2600-\u27BF\U0001F000-\U0001FAFF]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

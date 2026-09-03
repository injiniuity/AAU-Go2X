import json
from pathlib import Path
from typing import Any

from .. import config
from ..utils.logging_utils import print
from ..utils.text_utils import build_room_lookup, normalize_name_key


class MapService:
    def __init__(self, *, state: Any, name_matcher: Any) -> None:
        self._state = state
        self._name_matcher = name_matcher
        self.room_by_location_key = build_room_lookup(config.ROOM_ENTRY_CONFIG)

    def resolve_office_file(self) -> Path | None:
        for candidate in config.OFFICE210_FILE_CANDIDATES:
            if candidate.exists():
                return candidate
        return None

    def load_active_map_id(self) -> None:
        office_file = self.resolve_office_file()
        if office_file is not None:
            try:
                with office_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                map_data = data.get("map", {}) if isinstance(data, dict) else {}
                self._state.active_map_id = str(map_data.get("id", "")).strip()
                if self._state.active_map_id:
                    print(f"[SLAM] Active map id loaded: {self._state.active_map_id} ({office_file})", flush=True)
                    return
                print(f"[SLAM] office map file has no map id: {office_file}", flush=True)
            except Exception as exc:
                print(f"[SLAM] Failed to load map id from office map file: {exc}", flush=True)

        self._state.active_map_id = ""
        print("[SLAM] map id not found; it will not be set automatically", flush=True)

    def load_named_locations(self) -> None:
        office_file = self.resolve_office_file()
        if office_file is None:
            self._state.named_locations = {}
            self._state.initial_pose = None
            self._name_matcher.clear_cache()
            print(
                "[SLAM] office210.json not found in known paths: "
                + ", ".join(str(path) for path in config.OFFICE210_FILE_CANDIDATES),
                flush=True,
            )
            return

        try:
            with office_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            poses = data.get("poses", []) if isinstance(data, dict) else []
            loaded = {}
            init_pose = None

            for index, pose in enumerate(poses):
                explicit_kind = str(pose.get("kind", "")).strip()
                fallback_name = (
                    config.DEFAULT_OFFICE210_NAMES[index]
                    if index < len(config.DEFAULT_OFFICE210_NAMES)
                    else f"point{index}"
                )
                name = explicit_kind if explicit_kind and explicit_kind != "current_pose" else fallback_name
                entry = {
                    "name": name,
                    "x": float(pose["x"]),
                    "y": float(pose["y"]),
                    "yaw": float(pose.get("yaw", 0.0)),
                }
                if index == 0 or normalize_name_key(name) in {"initial", "initail"}:
                    init_pose = entry
                else:
                    loaded[normalize_name_key(name)] = entry

            self._state.named_locations = loaded
            self._state.initial_pose = init_pose
            self._name_matcher.clear_cache()
            print(f"[SLAM] Loaded {len(self._state.named_locations)} named locations from {office_file}", flush=True)
        except Exception as exc:
            self._state.named_locations = {}
            self._state.initial_pose = None
            self._name_matcher.clear_cache()
            print(f"[SLAM] Failed to load office210 locations: {exc}", flush=True)

    def match_location(self, location: str):
        if not self._state.named_locations:
            self.load_named_locations()
        key = normalize_name_key(location)
        if not key or not self._state.named_locations:
            return None
        if key in self._state.named_locations:
            return self._state.named_locations[key]
        return self._name_matcher.match_location(location)

    def match_location_exact(self, location: str):
        if not self._state.named_locations:
            self.load_named_locations()
        key = normalize_name_key(location)
        if not key or not self._state.named_locations:
            return None
        return self._state.named_locations.get(key)

    def match_first_location(self, *location_names: str):
        for location_name in location_names:
            pose = self.match_location(location_name)
            if pose is not None:
                return pose
        return None

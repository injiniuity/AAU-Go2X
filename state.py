import asyncio
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class RuntimeState:
    no_robot: bool = False
    text_mode: bool = False
    conn: Any = None
    audio_track: Any = None
    audio_hub: Any = None
    speak_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    stop_event: threading.Event = field(default_factory=threading.Event)
    command_queue: Any = None
    assistant_busy: threading.Event = field(default_factory=threading.Event)
    tts_active: threading.Event = field(default_factory=threading.Event)
    wakeword_ignore_until: float = 0.0
    passive_listen_queue: Any = None
    passive_listen_deadline: float = 0.0
    text_mode_prompt_queue: Any = None
    text_mode_prompt_label: str = ""
    camera_preview_task: asyncio.Task | None = None

    named_locations: dict[str, dict[str, Any]] = field(default_factory=dict)
    initial_pose: dict[str, Any] | None = None
    active_map_id: str = ""
    current_pose: dict[str, float] = field(
        default_factory=lambda: {"x": 0.0, "y": 0.0, "yaw": 0.0}
    )
    last_pose_update_monotonic: float = 0.0
    nav_state: str | None = None
    nav_state_seq: int = 0
    localization_init_state: str = "unknown"
    localization_init_waiters: list[tuple[Any, Any]] = field(default_factory=list)
    audiohub_play_state: dict[str, Any] = field(
        default_factory=lambda: {
            "is_playing": False,
            "current_audio_unique_id": "",
            "current_audio_custom_name": "",
        }
    )
    audiohub_play_waiters: list[tuple[Any, Any, Any]] = field(default_factory=list)
    slam_response_waiters: dict[str, list[tuple[Any, Any]]] = field(default_factory=dict)

    name_matcher_model: Any = None
    name_matcher_embeddings: dict[str, np.ndarray] = field(default_factory=dict)
    name_matcher_transforms: dict[str, str] = field(default_factory=dict)
    name_matcher_ready: bool = False
    name_matcher_failed: bool = False
    arpabet_predictor: Any = None
    arpabet_cache: dict[str, str] = field(default_factory=dict)

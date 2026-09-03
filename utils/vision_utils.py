"""Stateless image helpers for the Go2 camera pipeline."""

import base64
import re
from pathlib import Path

import cv2
import numpy as np


def encode_frame_as_jpeg_data_url(frame_bgr: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".jpg", frame_bgr)
    if not ok:
        raise RuntimeError("Failed to encode camera frame as JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def sanitize_capture_name(text: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", text.strip())
    return safe.strip("_") or "unknown"


def save_frame(frame_bgr: np.ndarray, capture_dir: Path, filename: str) -> str:
    capture_dir.mkdir(parents=True, exist_ok=True)
    file_path = capture_dir / filename
    if not cv2.imwrite(str(file_path), frame_bgr):
        raise RuntimeError(f"Failed to save VLM frame to {file_path}")
    return str(file_path)

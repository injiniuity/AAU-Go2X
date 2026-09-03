#!/usr/bin/env python3
"""Run YOLO on the Go2 front camera stream and highlight person/chair detections."""

import argparse
import asyncio
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

def find_local_webrtc_source() -> Path:
    """Find an adjacent clone of the external WebRTC dependency."""
    for directory in Path(__file__).resolve().parents:
        candidate = directory / "unitree_webrtc_connect"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not find unitree_webrtc_connect. Clone it next to this project or one of its parent folders."
    )


LOCAL_WEBRTC_SRC = find_local_webrtc_source()
if str(LOCAL_WEBRTC_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_WEBRTC_SRC))

from unitree_webrtc_connect.webrtc_driver import (
    UnitreeWebRTCConnection,
    WebRTCConnectionMethod,
)

logging.getLogger("aiortc.codecs.h264").setLevel(logging.ERROR)

DEFAULT_ROBOT_IP = os.environ.get("UNITREE_ROBOT_IP", "192.168.88.154")
DEFAULT_AES_KEY = os.environ.get("UNITREE_AES_128_KEY")
WINDOW_NAME = "Go2 YOLO Detection"
BOX_COLOR = (60, 220, 60)
TEXT_COLOR = (255, 255, 255)

latest_video_frame: Optional[np.ndarray] = None
video_frame_lock = threading.Lock()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect people or chairs from the Go2 front camera stream.",
    )
    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP, help="Go2 STA IP address")
    parser.add_argument("--aes-key", default=DEFAULT_AES_KEY, help="AES-128 key if required")
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Ultralytics YOLO model path or name",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["person", "chair"],
        help="COCO class names to keep",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Confidence threshold",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Ultralytics device string, e.g. cpu, 0",
    )
    return parser.parse_args()


def build_connection(robot_ip: str, aes_key: Optional[str]) -> UnitreeWebRTCConnection:
    kwargs = {"ip": robot_ip}
    if aes_key:
        kwargs["aes_128_key"] = aes_key
    return UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, **kwargs)


async def recv_camera_stream(track) -> None:
    global latest_video_frame
    while True:
        frame = await track.recv()
        img = frame.to_ndarray(format="bgr24")
        with video_frame_lock:
            latest_video_frame = img.copy()


def resolve_target_class_ids(model: YOLO, class_names: Sequence[str]) -> Tuple[Set[int], Dict[int, str]]:
    names = model.names
    lowered = {str(name).lower(): idx for idx, name in names.items()}
    selected_ids: Set[int] = set()
    missing: List[str] = []
    for class_name in class_names:
        idx = lowered.get(class_name.lower())
        if idx is None:
            missing.append(class_name)
            continue
        selected_ids.add(idx)
    if missing:
        raise ValueError(f"Unknown model classes: {', '.join(missing)}")
    selected_names = {idx: str(names[idx]) for idx in selected_ids}
    return selected_ids, selected_names


def annotate_frame(
    frame: np.ndarray,
    detections,
    selected_ids: Set[int],
    selected_names: Dict[int, str],
) -> Tuple[np.ndarray, List[str]]:
    annotated = frame.copy()
    summary: List[str] = []

    if detections.boxes is None:
        return annotated, summary

    boxes = detections.boxes.xyxy.cpu().numpy().astype(int)
    confs = detections.boxes.conf.cpu().numpy()
    classes = detections.boxes.cls.cpu().numpy().astype(int)

    for box, conf, cls_id in zip(boxes, confs, classes):
        if cls_id not in selected_ids:
            continue
        x1, y1, x2, y2 = box.tolist()
        label = f"{selected_names[cls_id]} {conf:.2f}"
        summary.append(label)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), BOX_COLOR, 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            TEXT_COLOR,
            2,
            cv2.LINE_AA,
        )
    return annotated, summary


async def main() -> None:
    args = parse_args()

    model = YOLO(args.model)
    selected_ids, selected_names = resolve_target_class_ids(model, args.classes)

    conn = build_connection(args.robot_ip, args.aes_key)
    await conn.connect()
    conn.video.add_track_callback(recv_camera_stream)
    conn.video.switchVideoChannel(True)

    print(f"Connected to Go2 camera at {args.robot_ip}")
    print(f"Detecting classes: {', '.join(selected_names.values())}")
    print("Press Q to quit.")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while True:
            with video_frame_lock:
                frame = None if latest_video_frame is None else latest_video_frame.copy()

            if frame is None:
                preview = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(
                    preview,
                    "Waiting for camera frames...",
                    (40, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            else:
                results = model.predict(
                    source=frame,
                    conf=args.conf,
                    imgsz=args.imgsz,
                    device=args.device,
                    verbose=False,
                )
                preview, summary = annotate_frame(frame, results[0], selected_ids, selected_names)
                status = "Detected: " + (", ".join(summary) if summary else "none")
                cv2.putText(
                    preview,
                    status,
                    (20, preview.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow(WINDOW_NAME, preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            await asyncio.sleep(0.01)
    finally:
        cv2.destroyAllWindows()
        await conn.disconnect()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)

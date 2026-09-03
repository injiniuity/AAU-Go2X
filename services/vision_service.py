import asyncio
import threading
from typing import Any

import cv2
import numpy as np


class VisionService:
    def __init__(
        self,
        *,
        yolo_cls: Any,
        yolo_model_name: str,
        yolo_target_classes: tuple[str, ...],
    ) -> None:
        self._yolo_cls = yolo_cls
        self._yolo_model_name = yolo_model_name
        self._yolo_target_classes = yolo_target_classes
        self._latest_frame: np.ndarray | None = None
        self._frame_lock = threading.Lock()
        self._frame_updated = threading.Event()
        self._reader_task: asyncio.Task | None = None
        self._find_person_yolo_model = None

    async def _video_reader_loop(self, track) -> None:
        try:
            while True:
                frame = await track.recv()
                img = frame.to_ndarray(format="bgr24")
                with self._frame_lock:
                    self._latest_frame = img
                self._frame_updated.set()
        except Exception as exc:
            if exc.__class__.__name__ != "MediaStreamError":
                raise

    async def recv_camera_stream(self, track) -> None:
        if self._reader_task is None or self._reader_task.done():
            self._reader_task = asyncio.create_task(self._video_reader_loop(track))

    def _get_find_person_yolo_model(self):
        if self._yolo_cls is None:
            return None
        if self._find_person_yolo_model is None:
            self._find_person_yolo_model = self._yolo_cls(self._yolo_model_name)
        return self._find_person_yolo_model

    def detect_find_person_objects(
        self,
        frame_bgr: np.ndarray,
    ) -> tuple[list[tuple[list[int], float, str]], dict[str, int]]:
        model = self._get_find_person_yolo_model()
        if model is None:
            return [], {}

        names = model.names
        target_ids = {
            idx
            for idx, name in names.items()
            if str(name).lower() in self._yolo_target_classes
        }
        if not target_ids:
            return [], {}

        try:
            results = model.predict(source=frame_bgr, conf=0.30, imgsz=640, verbose=False)
        except Exception as exc:
            print(f"[find_person] YOLO inference failed: {exc}", flush=True)
            return [], {}

        if not results:
            return [], {}

        boxes = results[0].boxes
        if boxes is None:
            return [], {}

        xyxy = boxes.xyxy.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)

        detections: list[tuple[list[int], float, str]] = []
        counts: dict[str, int] = {}
        for box, conf, class_id in zip(xyxy, confs, class_ids):
            if class_id not in target_ids:
                continue
            class_name = str(names[class_id]).lower()
            detections.append((box.tolist(), float(conf), class_name))
            counts[class_name] = counts.get(class_name, 0) + 1

        return detections, counts

    def draw_find_person_boxes(self, frame_bgr: np.ndarray) -> np.ndarray:
        annotated = frame_bgr.copy()
        detections, _ = self.detect_find_person_objects(frame_bgr)
        for box, conf, class_name in detections:
            x1, y1, x2, y2 = box
            label = f"{class_name} {conf:.2f}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                annotated,
                label,
                (x1, max(24, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        return annotated

    async def get_latest_camera_frame(self, timeout: float = 5.0) -> np.ndarray:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            with self._frame_lock:
                if self._latest_frame is not None:
                    return self._latest_frame.copy()
            remaining = max(0.0, deadline - loop.time())
            if remaining == 0:
                break
            await asyncio.to_thread(self._frame_updated.wait, min(0.25, remaining))
        raise TimeoutError("No camera frame received from the robot")

    def build_preview_frame(self) -> np.ndarray:
        with self._frame_lock:
            frame = None if self._latest_frame is None else self._latest_frame.copy()
        if frame is not None:
            return frame
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(
            frame,
            "Waiting for camera frames...",
            (40, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return frame

    async def camera_preview_loop(self) -> None:
        cv2.namedWindow("Go2 Camera", cv2.WINDOW_NORMAL)
        try:
            while True:
                frame = self.build_preview_frame()
                cv2.imshow("Go2 Camera", frame)
                cv2.waitKey(1)
                await asyncio.sleep(0.03)
        except asyncio.CancelledError:
            raise
        finally:
            cv2.destroyWindow("Go2 Camera")

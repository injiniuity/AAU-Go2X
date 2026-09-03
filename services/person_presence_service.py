import asyncio
import json
import time
from pathlib import Path

from .. import config
from ..utils.logging_utils import print
from ..utils.vision_utils import (
    encode_frame_as_jpeg_data_url,
    sanitize_capture_name,
    save_frame,
)


PERSON_PRESENCE_SYSTEM_PROMPT = """You determine whether a person is visible near the seat in front of the robot.
Use both images. Any visible body part counts as present. Only choose absent when neither image
contains a person. Reply with JSON only: {"presence":"present|absent","reason":"short reason"}."""


def build_person_presence_prompt(description: str) -> str:
    return (
        f"Check whether a person is visible around {description}'s seat. "
        "The first image is standing height and the second is sitting height. "
        "Red boxes are detector hints, not the final decision. Return JSON only."
    )


class PersonPresenceService:
    """Captures two viewpoints and determines whether a seat is occupied."""

    def __init__(self, *, state, connection, navigation, vision, complete_chat) -> None:
        self.state = state
        self.connection = connection
        self.navigation = navigation
        self.vision = vision
        self.complete_chat = complete_chat

    @staticmethod
    def resolve_presence(raw_answer: str, person_count: int) -> tuple[str, str]:
        """Prefer local detection, then the VLM's required JSON, otherwise stay uncertain."""
        if person_count > 0:
            return "present", "A person was detected in the camera view near the seat."
        try:
            payload = json.loads(raw_answer)
        except (TypeError, json.JSONDecodeError):
            return "unknown", "The vision model did not return a valid presence result."

        if not isinstance(payload, dict):
            return "unknown", "The vision model returned an unexpected result."
        presence = str(payload.get("presence", "")).strip().casefold()
        reason = str(payload.get("reason", "")).strip()
        if presence not in {"present", "absent"}:
            return "unknown", reason or "The vision model did not provide a valid presence value."
        return presence, reason or f"The vision model judged the seat {presence}."

    async def find_person(self, description: str, do_skill) -> str:
        if self.state.no_robot:
            return json.dumps({"status": "ok", "description": description, "presence": "unknown", "note": "no-robot mode"})
        if not await self.connection.ensure_robot_connection():
            return json.dumps({"status": "error", "message": "Robot connection lost"})

        started_at = time.monotonic()

        def log(stage: str) -> None:
            print(f"[find_person +{time.monotonic() - started_at:0.2f}s] {stage}", flush=True)

        sit_result = None
        saved_before_path = ""
        saved_after_path = ""
        try:
            self.navigation.send_slam_cmd("navigation/stop")
            await asyncio.sleep(1.0)
            await do_skill("StandUp")
            await asyncio.sleep(2.0)

            frame_before = await self.vision.get_latest_camera_frame()
            _, before_counts = self.vision.detect_find_person_objects(frame_before)
            before_image = self.vision.draw_find_person_boxes(frame_before)
            capture_dir = Path(config.SCRIPT_DIR) / "find_person_vlm_captures" / (
                f"{time.strftime('%Y%m%d_%H%M%S')}_{sanitize_capture_name(description)}"
            )
            saved_before_path = save_frame(before_image, capture_dir, "before_sit_vlm.jpg")

            sit_result = json.loads(await do_skill("Sit"))
            await asyncio.sleep(5.0)

            frame_after = await self.vision.get_latest_camera_frame()
            _, after_counts = self.vision.detect_find_person_objects(frame_after)
            after_image = self.vision.draw_find_person_boxes(frame_after)
            saved_after_path = save_frame(after_image, capture_dir, "after_sit_vlm.jpg")

            response = await self.complete_chat(
                model=config.VLM_MODEL,
                messages=[
                    {"role": "system", "content": PERSON_PRESENCE_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": build_person_presence_prompt(description)},
                            {"type": "image_url", "image_url": {"url": encode_frame_as_jpeg_data_url(before_image)}},
                            {"type": "image_url", "image_url": {"url": encode_frame_as_jpeg_data_url(after_image)}},
                        ],
                    },
                ],
            )
            raw_answer = response.choices[0].message.content.strip()
            person_count = before_counts.get("person", 0) + after_counts.get("person", 0)
            presence, reason = self.resolve_presence(raw_answer, person_count)
            answer = f"{description} appears to be at the seat." if presence == "present" else (
                f"{description} does not appear to be at the seat." if presence == "absent" else f"I could not reliably determine whether {description} is at the seat."
            )
            log(f"presence={presence} yolo_people={person_count}")
            return json.dumps({
                "status": "ok",
                "description": description,
                "presence": presence,
                "answer": answer,
                "reason": reason,
                "vlm_raw_answer": raw_answer,
                "sit": sit_result,
                "saved_images": {"before_sit": saved_before_path, "after_sit": saved_after_path},
            })
        except Exception as error:
            log(f"exception={error}")
            return json.dumps({"status": "error", "message": str(error), "sit": sit_result})
        finally:
            try:
                await do_skill("StandUp")
            except Exception as error:
                log(f"StandUp failed: {error}")
            try:
                self.navigation.send_slam_cmd("navigation/start")
            except Exception as error:
                log(f"navigation restart failed: {error}")

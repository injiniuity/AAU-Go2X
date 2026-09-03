import json
from typing import Optional

from .. import config
from ..utils.logging_utils import print
from ..utils.text_utils import normalize_name_key
from ..utils.vision_utils import encode_frame_as_jpeg_data_url

from unitree_webrtc_connect.constants import RTC_TOPIC, SPORT_CMD


class AssistantActions:
    """Implements the callable actions exposed to the LLM."""

    def __init__(self, *, state, connection, navigation, map_service, vision, audio, door_entry, person_presence, complete_chat) -> None:
        self.state = state
        self.connection = connection
        self.navigation = navigation
        self.map_service = map_service
        self.vision = vision
        self.audio = audio
        self.door_entry = door_entry
        self.person_presence = person_presence
        self.complete_chat = complete_chat

    async def return_to_saved_target(self, target: dict) -> dict:
        target_type = target.get("type")
        if target_type == "named_location":
            location = str(target.get("location", "")).strip()
            if not location:
                return {"status": "error", "message": "Saved return location was empty."}
            return json.loads(await self.go_to(location))

        if target_type == "pose":
            return await self.navigation.navigate_to_pose(
                {
                    "name": str(target.get("location", "starting position")),
                    "x": float(target["x"]),
                    "y": float(target["y"]),
                    "yaw": float(target.get("yaw", 0.0)),
                }
            )

        return {"status": "error", "message": "Unknown saved return target."}


    def build_seat_report_message(self, person: str, presence: str) -> str:
        seat_name = f"{person}'s seat"
        if presence == "present":
            return f"{person} appears to be at the seat."
        if presence == "absent":
            return f"{seat_name} appears empty."
        return f"It is unclear whether {person} is at the seat."


    async def do_skill(self, skill: str) -> str:
        if skill not in config.SKILLS:
            return json.dumps({"status": "error", "message": f"Unknown skill: {skill}"})
        if self.state.no_robot:
            return json.dumps({"status": "ok", "skill": skill, "note": "no-robot mode"})
        if not await self.connection.ensure_robot_connection():
            return json.dumps({"status": "error", "message": "Robot connection lost"})
        try:
            payload = {"api_id": SPORT_CMD[skill]}
            resp = await self.state.conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["SPORT_MOD"],
                payload,
            )
            code = resp.get("data", {}).get("header", {}).get("status", {}).get("code", -1)
            if code == 0:
                return json.dumps({"status": "ok", "skill": skill})
            return json.dumps({"status": "error", "code": code})
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})


    async def describe_view(self, question: str = "What do you see?") -> str:
        if self.state.no_robot:
            return json.dumps(
                {
                    "status": "not_implemented",
                    "message": "Vision is unavailable in no-robot mode.",
                }
            )
        if not await self.connection.ensure_robot_connection():
            return json.dumps({"status": "error", "message": "Robot connection lost"})
        try:
            frame = await self.vision.get_latest_camera_frame()
            image_url = encode_frame_as_jpeg_data_url(frame)
            response = await self.complete_chat(
                model=config.VLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are the visual perception module for a Unitree Go2 robot. "
                            "Answer only from the image. Keep answers short, clear, and natural."
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": question},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    },
                ],
            )
            answer = response.choices[0].message.content.strip()
            return json.dumps({"status": "ok", "answer": answer})
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})


    async def go_to(self, location: str) -> str:
        try:
            location_text = str(location).strip()
            location_key = normalize_name_key(location_text)

            pose = self.map_service.match_location(location_text)
            if pose is None:
                known = [v["name"] for v in self.state.named_locations.values()]
                return json.dumps(
                    {
                        "status": "error",
                        "message": f"Unknown location '{location_text}'. Known locations: {', '.join(known) if known else 'none'}",
                    }
                )
            if self.state.no_robot:
                return json.dumps(
                    {
                        "status": "ok",
                        "location": pose.get("name", location_text),
                        "note": "no-robot mode",
                    }
                )
            if not await self.connection.ensure_robot_connection():
                return json.dumps({"status": "error", "message": "Robot connection lost"})
            corridor_targets = self.navigation.build_corridor_transition_sequence(pose)
            if corridor_targets:
                step_results = []
                for target_pose in corridor_targets:
                    nav_result = await self.navigation.navigate_to_pose(target_pose)
                    step_results.append(
                        {
                            "target": target_pose.get("name", target_pose.get("kind", location_text)),
                            "result": nav_result,
                        }
                    )
                    if nav_result.get("status") != "ok":
                        return json.dumps(
                            {
                                "status": "error",
                                "location": pose.get("name", location_text),
                                "steps": step_results,
                                "message": nav_result.get("message", "Failed while traversing corridor."),
                            }
                        )
            if location_key.startswith("door"):
                approach_targets = self.navigation.build_door_approach_sequence(location_text)
                if approach_targets:
                    last_result = None
                    step_results = []
                    for target_pose in approach_targets:
                        last_result = await self.navigation.navigate_to_pose(target_pose)
                        step_results.append(
                            {
                                "target": target_pose.get("name", target_pose.get("kind", location_text)),
                                "result": last_result,
                            }
                        )
                        if last_result.get("status") != "ok":
                            return json.dumps(
                                {
                                    "status": "error",
                                    "location": pose.get("name", location_text),
                                    "steps": step_results,
                                    "message": last_result.get("message", "Failed while approaching door."),
                                }
                            )
                    if last_result is not None:
                        last_result["steps"] = step_results
                        return json.dumps(last_result)
            entry_plan = self.door_entry.resolve_room_entry_plan(pose.get("name", location_text))
            if entry_plan is not None:
                return_target = self.navigation.capture_return_target()
                result = await self.door_entry.handle_room_entry_navigation(
                    pose,
                    entry_plan,
                    return_target,
                    do_skill=self.do_skill,
                    return_to_saved_target=self.return_to_saved_target,
                    say_message=self.say_message,
                )
            else:
                result = await self.navigation.navigate_to_pose(pose)
            return json.dumps(result)
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})


    async def check_seat_and_report_back(self, person: str) -> str:
        if not await self.connection.ensure_robot_connection():
            return json.dumps({"status": "error", "message": "Robot connection lost"})

        return_target = self.navigation.capture_return_target()
        if return_target is None:
            return json.dumps(
                {
                    "status": "error",
                    "message": "Could not determine where to return after checking the seat.",
                }
            )

        go_result = json.loads(await self.go_to(person))
        if go_result.get("status") != "ok":
            return json.dumps(
                {
                    "status": "error",
                    "step": "go_to_target",
                    "target_person": person,
                    "result": go_result,
                }
            )

        matched_person = str(go_result.get("location", person)).strip() or person

        find_result = json.loads(await self.person_presence.find_person(matched_person, self.do_skill))
        if find_result.get("status") != "ok":
            return json.dumps(
                {
                    "status": "error",
                    "step": "find_person",
                    "target_person": matched_person,
                    "requested_person": person,
                    "result": find_result,
                }
            )

        return_result = await self.return_to_saved_target(return_target)
        if return_result.get("status") != "ok":
            return json.dumps(
                {
                    "status": "error",
                    "step": "return_to_start",
                    "target_person": matched_person,
                    "requested_person": person,
                    "return_target": return_target,
                    "find_result": find_result,
                    "result": return_result,
                }
            )

        report_message = self.build_seat_report_message(
            matched_person,
            str(find_result.get("presence", "uncertain")),
        )
        say_result = json.loads(await self.say_message(report_message))
        return json.dumps(
            {
                "status": "ok",
                "spoken": say_result.get("spoken") is True,
                "target_person": matched_person,
                "requested_person": person,
                "presence": find_result.get("presence", "uncertain"),
                "report": report_message,
                "return_target": return_target,
                "go_result": go_result,
                "find_result": find_result,
                "return_result": return_result,
                "say_result": say_result,
            }
        )


    async def deliver_message_to_person(self, 
        person: str,
        message: str,
        skill: Optional[str] = None,
    ) -> str:
        target_person = str(person).strip()
        spoken_text = str(message).strip()
        if not target_person:
            return json.dumps({"status": "error", "message": "Target person was empty."})
        if not spoken_text:
            return json.dumps({"status": "error", "message": "Message was empty."})

        go_result = json.loads(await self.go_to(target_person))
        if go_result.get("status") != "ok":
            return json.dumps(
                {
                    "status": "error",
                    "step": "go_to_target",
                    "target_person": target_person,
                    "result": go_result,
                }
            )

        matched_person = str(go_result.get("location", target_person)).strip() or target_person
        say_result = json.loads(await self.say_message(spoken_text, skill=skill))
        if say_result.get("status") != "ok":
            return json.dumps(
                {
                    "status": "error",
                    "step": "say_message",
                    "target_person": matched_person,
                    "requested_person": target_person,
                    "go_result": go_result,
                    "result": say_result,
                }
            )

        return json.dumps(
            {
                "status": "ok",
                "spoken": say_result.get("spoken") is True,
                "target_person": matched_person,
                "requested_person": target_person,
                "message": spoken_text,
                "skill": skill,
                "go_result": go_result,
                "say_result": say_result,
            }
        )


    async def say_message(self, message: str, skill: Optional[str] = None) -> str:
        spoken_text = message.strip()
        if not spoken_text:
            return json.dumps({"status": "error", "message": "Message was empty."})
        skill_result = None
        try:
            if skill:
                if skill not in config.SKILLS:
                    return json.dumps({"status": "error", "message": f"Unknown skill: {skill}"})
                skill_result = json.loads(await self.do_skill(skill))
            print(f"[Say] {spoken_text}")
            spoken = await self.audio.speak(spoken_text)
            return json.dumps(
                {
                    "status": "ok",
                    "spoken": spoken,
                    "message": spoken_text,
                    "skill": skill,
                    "skill_result": skill_result,
                }
            )
        except Exception as e:
            return json.dumps(
                {
                    "status": "error",
                    "message": str(e),
                    "spoken": False,
                    "skill": skill,
                    "skill_result": skill_result,
                }
            )

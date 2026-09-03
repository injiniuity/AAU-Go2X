import asyncio
import json
import re
import time
from typing import Optional

from .. import config
from ..utils.logging_utils import print
from ..utils.text_utils import normalize_name_key


class DoorEntryService:
    """Plans room entry, obtains permission, and performs door navigation."""

    def __init__(self, *, state, map_service, navigation, audio, input_service, complete_chat) -> None:
        self.state = state
        self.map_service = map_service
        self.navigation = navigation
        self.audio = audio
        self.input_service = input_service
        self.complete_chat = complete_chat

    def get_room_id_for_location(self, location_name: str) -> Optional[str]:
        return ROOM_BY_LOCATION_KEY.get(normalize_name_key(location_name))


    def get_current_room_id(self, ) -> Optional[str]:
        nearest_pose, _ = self.navigation.get_nearest_named_location()
        if nearest_pose is not None:
            nearest_name = str(nearest_pose.get("name", "")).strip()
            if normalize_name_key(nearest_name) == "corridor":
                return "corridor"
            room_id = self.get_room_id_for_location(nearest_name)
            if room_id:
                return room_id
        return None


    def resolve_room_entry_plan(self, destination_name: str) -> Optional[dict]:
        destination_key = normalize_name_key(destination_name)
        if destination_key in {"initial", "initail"}:
            return None

        current_room = self.get_current_room_id()
        destination_room = self.get_room_id_for_location(destination_name)
        if destination_room is None:
            return None

        room_config = config.ROOM_ENTRY_CONFIG.get(destination_room, {})
        door_names = list(room_config.get("doors", []))
        if not door_names:
            return None

        transition_override = config.ROOM_TRANSITION_DOOR_OVERRIDES.get((current_room, destination_room))
        if transition_override:
            door_names = [str(name).strip() for name in transition_override if str(name).strip()]

        member_door_lookup = {
            normalize_name_key(name): str(door_name).strip()
            for name, door_name in room_config.get("member_doors", {}).items()
            if str(name).strip() and str(door_name).strip()
        }
        selected_door_name = member_door_lookup.get(destination_key)
        if selected_door_name and not transition_override:
            door_names = [selected_door_name]

        if destination_key in {normalize_name_key(name) for name in door_names}:
            return None

        if current_room == destination_room:
            return None

        print(
            f"[DoorPlan] current_room={current_room or 'outside'} "
            f"destination_room={destination_room} via {', '.join(door_names)}",
            flush=True,
        )

        return {
            "room_id": destination_room,
            "door_names": door_names,
            "current_room": current_room,
        }


    def classify_entry_permission_fallback(self, reply_text: str) -> str:
        normalized = normalize_name_key(reply_text)
        if not normalized:
            return "unknown"
        normalized = re.sub(r"[^0-9a-zA-Z\uac00-\ud7a3\s]", " ", normalized)
        normalized = " ".join(normalized.split())


        if normalized in {"no", "nope", "nah"}:
            return "denied"
        if normalized in {"yes", "yeah", "yep", "sure", "okay", "ok"}:
            return "allowed"
        return "unknown"


    async def classify_entry_permission(self, reply_text: str) -> str:
        normalized = normalize_name_key(reply_text)
        if not normalized:
            return "unknown"

        fallback_result = self.classify_entry_permission_fallback(reply_text)
        try:
            response = await self.complete_chat(
                model=config.LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You classify whether a short reply to 'Knock knock, can I come in?' "
                            "means permission is granted, denied, or unclear. "
                            "Reply with exactly one word: allowed, denied, or unknown."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Reply: {reply_text}",
                    },
                ],
                temperature=0,
            )
            verdict = (
                response.choices[0].message.content.strip().casefold()
                if response.choices and response.choices[0].message.content
                else ""
            )
            if verdict in {"allowed", "denied", "unknown"}:
                return verdict
        except Exception as e:
            print(f"[Door] LLM permission classification failed: {e}", flush=True)

        return fallback_result


    async def listen_for_entry_response(self, timeout: float = config.ENTRY_PERMISSION_TIMEOUT_SEC) -> str:

        if self.state.text_mode:
            response_queue: asyncio.Queue = asyncio.Queue(maxsize=1)
            self.state.text_mode_prompt_queue = response_queue
            self.state.text_mode_prompt_label = "Door response"
            print("Door response: ", end="", flush=True)
            try:
                return str(await asyncio.wait_for(response_queue.get(), timeout=timeout)).strip()
            except asyncio.TimeoutError:
                print("\n[Door] Timed out waiting for typed reply", flush=True)
                return ""
            finally:
                if self.state.text_mode_prompt_queue is response_queue:
                    self.state.text_mode_prompt_queue = None
                    self.state.text_mode_prompt_label = ""

        response_queue: asyncio.Queue = asyncio.Queue(maxsize=1)
        self.state.passive_listen_queue = response_queue
        self.state.passive_listen_deadline = time.time() + timeout
        self.state.wakeword_ignore_until = 0.0
        print(f"[Door] Listening for a reply for up to {timeout:.1f}s", flush=True)

        try:
            audio_np = await asyncio.wait_for(response_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return ""
        finally:
            if self.state.passive_listen_queue is response_queue:
                self.state.passive_listen_queue = None
                self.state.passive_listen_deadline = 0.0

        print("[Door] Transcribing reply...", flush=True)
        try:
            return await asyncio.to_thread(self.input_service.transcribe, audio_np)
        except Exception as e:
            print(f"[Door] Failed to transcribe reply: {e}", flush=True)
            return ""


    async def handle_room_entry_navigation(self, destination_pose: dict, entry_plan: dict, return_target: Optional[dict], *, do_skill, return_to_saved_target, say_message) -> dict:
        room_id = str(entry_plan.get("room_id", "")).strip() or "unknown"
        door_names = [str(name).strip() for name in entry_plan.get("door_names", []) if str(name).strip()]
        if not door_names:
            return {
                "status": "error",
                "message": f"No door points were configured for room {room_id}.",
                "destination": destination_pose.get("name", "target"),
            }
        if config.ROOM_ENTRY_REQUIRE_PERMISSION and return_target is None:
            return {
                "status": "error",
                "message": "Could not determine where to return if room entry is denied.",
                "room_id": room_id,
                "destination": destination_pose.get("name", "target"),
            }

        resolved_doors: list[tuple[str, list[dict]]] = []
        missing_door_names: list[str] = []
        for door_name in door_names:
            approach_targets = self.navigation.build_door_approach_sequence(door_name)
            if approach_targets:
                resolved_doors.append((door_name, approach_targets))
            else:
                missing_door_names.append(door_name)

        if not resolved_doors:
            print(
                "[Door] No configured door points were found in the loaded map; "
                "skipping permission check and heading directly to destination",
                flush=True,
            )
            destination_result = await self.navigation.navigate_to_pose(destination_pose)
            if destination_result.get("status") != "ok":
                return {
                    "status": "error",
                    "step": "go_to_destination_without_door_points",
                    "room_id": room_id,
                    "destination": destination_pose.get("name", "target"),
                    "missing_doors": missing_door_names,
                    "permission": "skipped_missing_door_points",
                    "result": destination_result,
                }
            return {
                "status": "ok",
                "location": destination_result.get("location", destination_pose.get("name", "target")),
                "x": destination_result.get("x"),
                "y": destination_result.get("y"),
                "yaw": destination_result.get("yaw"),
                "room_id": room_id,
                "door_results": [],
                "missing_doors": missing_door_names,
                "reply_text": "",
                "permission": "skipped_missing_door_points",
                "thank_you_result": None,
                "post_permission_results": [],
            }

        if missing_door_names:
            print(
                "[Door] Missing door points in the loaded map: "
                + ", ".join(missing_door_names)
                + " -- using available door points only",
                flush=True,
            )

        door_results = []
        last_door_name = ""
        for door_name, approach_targets in resolved_doors:
            last_door_name = door_name
            per_door_steps = []
            for target_pose in approach_targets:
                nav_result = await self.navigation.navigate_to_pose(target_pose)
                per_door_steps.append(
                    {
                        "target": target_pose.get("name", target_pose.get("kind", door_name)),
                        "result": nav_result,
                    }
                )
                if nav_result.get("status") != "ok":
                    door_results.append({"door": door_name, "steps": per_door_steps})
                    return {
                        "status": "error",
                        "step": "go_to_door",
                        "room_id": room_id,
                        "door": door_name,
                        "destination": destination_pose.get("name", "target"),
                        "door_results": door_results,
                        "result": nav_result,
                    }
            door_results.append({"door": door_name, "steps": per_door_steps})

        if not config.ROOM_ENTRY_REQUIRE_PERMISSION:
            print("[Door] Permission check skipped; continuing directly to destination", flush=True)
            post_permission_results = []
            post_permission_sequence = self.navigation.build_door_post_permission_sequence(last_door_name)
            if not post_permission_sequence:
                print("[Door] No post-door waypoint configured; heading directly to destination", flush=True)
            for target_pose in post_permission_sequence:
                print(
                    f"[Door] Moving to post-door waypoint: "
                    f"{target_pose.get('name', target_pose.get('kind', last_door_name))}",
                    flush=True,
                )
                nav_result = await self.navigation.navigate_to_pose(target_pose)
                post_permission_results.append(
                    {
                        "target": target_pose.get("name", target_pose.get("kind", last_door_name)),
                        "result": nav_result,
                    }
                )
                if nav_result.get("status") != "ok":
                    return {
                        "status": "error",
                        "step": "go_to_post_door_waypoint",
                        "room_id": room_id,
                        "destination": destination_pose.get("name", "target"),
                        "door_results": door_results,
                        "permission": "skipped",
                        "post_permission_results": post_permission_results,
                        "result": nav_result,
                    }

            print(
                f"[Door] Moving to final destination without permission check: "
                f"{destination_pose.get('name', 'target')}",
                flush=True,
            )
            destination_result = await self.navigation.navigate_to_pose(destination_pose)
            if destination_result.get("status") != "ok":
                return {
                    "status": "error",
                    "step": "go_to_destination_after_door",
                    "room_id": room_id,
                    "destination": destination_pose.get("name", "target"),
                    "door_results": door_results,
                    "permission": "skipped",
                    "post_permission_results": post_permission_results,
                    "result": destination_result,
                }
            return {
                "status": "ok",
                "location": destination_result.get("location", destination_pose.get("name", "target")),
                "x": destination_result.get("x"),
                "y": destination_result.get("y"),
                "yaw": destination_result.get("yaw"),
                "room_id": room_id,
                "door_results": door_results,
                "reply_text": "",
                "permission": "skipped",
                "thank_you_result": None,
                "post_permission_results": post_permission_results,
            }

        knock_result = await self.audio.play_knock_prompt_parallel(do_skill)
        if knock_result.get("status") != "ok":
            return {
                "status": "error",
                "step": (
                    "knock_gesture"
                    if knock_result.get("skill_result", {}).get("status") != "ok"
                    else "knock_and_request_entry"
                ),
                "room_id": room_id,
                "destination": destination_pose.get("name", "target"),
                "door_results": door_results,
                "result": knock_result,
            }

        reply_text = (await self.listen_for_entry_response()).strip()
        permission = await self.classify_entry_permission(reply_text)
        print(f"[Door] Reply='{reply_text}' permission={permission}", flush=True)

        if permission == "allowed":
            thank_you_result = await self.audio.play_thank_you_prompt()
            print("[Door] Entry allowed; starting post-permission navigation", flush=True)
            post_permission_results = []
            reverse_waypoint = (
                self.map_service.match_location_exact(f"{last_door_name}_waypoint1")
                or self.map_service.match_location_exact(f"{last_door_name}_waypoint")
            )
            if reverse_waypoint is not None:
                door_pose = self.map_service.match_location_exact(last_door_name)
                if door_pose is not None:
                    print(
                        f"[Door] Reversing away from {last_door_name} toward "
                        f"{reverse_waypoint.get('name', reverse_waypoint.get('kind', 'door_waypoint'))}",
                        flush=True,
                    )
                    reverse_result = await self.navigation.reverse_from_door_to_waypoint(
                        door_pose,
                        reverse_waypoint,
                    )
                    post_permission_results.append(
                        {
                            "target": reverse_waypoint.get("name", reverse_waypoint.get("kind", last_door_name)),
                            "result": reverse_result,
                        }
                    )
                    if reverse_result.get("status") != "ok":
                        return {
                            "status": "error",
                            "step": "reverse_after_entry_permission",
                            "room_id": room_id,
                            "destination": destination_pose.get("name", "target"),
                            "door_results": door_results,
                            "reply_text": reply_text,
                            "permission": permission,
                            "thank_you_result": thank_you_result,
                            "post_permission_results": post_permission_results,
                            "result": reverse_result,
                        }
            else:
                print(
                    f"[Door] No reverse waypoint configured for {last_door_name}; continuing without reverse",
                    flush=True,
                )
            post_permission_sequence = self.navigation.build_door_post_permission_sequence(last_door_name)
            if not post_permission_sequence:
                print("[Door] No post-permission waypoint configured; heading directly to destination", flush=True)
            for target_pose in post_permission_sequence:
                print(
                    f"[Door] Moving to post-permission waypoint: "
                    f"{target_pose.get('name', target_pose.get('kind', last_door_name))}",
                    flush=True,
                )
                nav_result = await self.navigation.navigate_to_pose(target_pose)
                post_permission_results.append(
                    {
                        "target": target_pose.get("name", target_pose.get("kind", last_door_name)),
                        "result": nav_result,
                    }
                )
                if nav_result.get("status") != "ok":
                    return {
                        "status": "error",
                        "step": "go_to_post_permission_waypoint",
                        "room_id": room_id,
                        "destination": destination_pose.get("name", "target"),
                        "door_results": door_results,
                        "reply_text": reply_text,
                        "permission": permission,
                        "thank_you_result": thank_you_result,
                        "post_permission_results": post_permission_results,
                        "result": nav_result,
                    }
            print(
                f"[Door] Moving to final destination after entry: "
                f"{destination_pose.get('name', 'target')}",
                flush=True,
            )
            destination_result = await self.navigation.navigate_to_pose(destination_pose)
            if destination_result.get("status") != "ok":
                return {
                    "status": "error",
                    "step": "go_to_destination_after_entry",
                    "room_id": room_id,
                    "destination": destination_pose.get("name", "target"),
                    "door_results": door_results,
                    "reply_text": reply_text,
                    "permission": permission,
                    "thank_you_result": thank_you_result,
                    "post_permission_results": post_permission_results,
                    "result": destination_result,
                }
            return {
                "status": "ok",
                "location": destination_result.get("location", destination_pose.get("name", "target")),
                "x": destination_result.get("x"),
                "y": destination_result.get("y"),
                "yaw": destination_result.get("yaw"),
                "room_id": room_id,
                "door_results": door_results,
                "reply_text": reply_text,
                "permission": permission,
                "thank_you_result": thank_you_result,
                "post_permission_results": post_permission_results,
            }

        if permission == "denied":
            report_message = "They said I can't come in."
        else:
            report_message = "No one said I could come in."

        return_result = await return_to_saved_target(return_target)
        say_result = json.loads(await say_message(report_message))
        return {
            "status": "error",
            "step": "entry_permission_denied",
            "room_id": room_id,
            "destination": destination_pose.get("name", "target"),
            "door_results": door_results,
            "reply_text": reply_text,
            "permission": permission,
            "report": report_message,
            "return_target": return_target,
            "return_result": return_result,
            "say_result": say_result,
        }

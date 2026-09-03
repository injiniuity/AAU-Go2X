import asyncio
import json
import math
import time
from typing import Optional

from .. import config
from ..utils.logging_utils import print
from ..utils.navigation_planner import door_approach_sequence, normalize_yaw, with_rotated_yaw
from ..utils.text_utils import normalize_name_key, quat_to_yaw

from unitree_webrtc_connect.constants import RTC_TOPIC, SPORT_CMD


class NavigationService:
    """Owns SLAM callbacks, localization lifecycle, and robot navigation."""

    def __init__(self, *, state, connection, map_service) -> None:
        self.state = state
        self.connection = connection
        self.map_service = map_service

    def pose_callback(self, msg):
        try:
            data = msg.get("data", msg)
            if isinstance(data, str):
                data = json.loads(data)
            pos = data.get("pose", {}).get("pose", {}).get("position", {})
            ori = data.get("pose", {}).get("pose", {}).get("orientation", {})
            self.state.current_pose["x"] = pos.get("x", 0.0)
            self.state.current_pose["y"] = pos.get("y", 0.0)
            self.state.current_pose["yaw"] = quat_to_yaw(
                ori.get("x", 0.0),
                ori.get("y", 0.0),
                ori.get("z", 0.0),
                ori.get("w", 1.0),
            )
            self.state.last_pose_update_monotonic = time.monotonic()
        except Exception as e:
            print(f"[SLAM Pose Error] {e}", flush=True)


    def resolve_slam_response_waiters(self, command_key: str, value: str) -> None:
        waiters = self.state.slam_response_waiters.pop(command_key, [])
        for loop, future in waiters:
            if future.done():
                continue
            loop.call_soon_threadsafe(future.set_result, value)


    def resolve_localization_init_waiters(self, ) -> None:
        remaining = []
        for loop, future in self.state.localization_init_waiters:
            if future.done():
                continue
            loop.call_soon_threadsafe(future.set_result, self.state.localization_init_state)
        self.state.localization_init_waiters = remaining


    def slam_server_log_callback(self, msg):
        data = msg.get("data", msg)
        if isinstance(data, str) and "navigation/state_transition/" in data:
            self.state.nav_state = data.rsplit("/", 1)[-1].strip()
            self.state.nav_state_seq += 1
            print(f"[SLAM Nav State] {self.state.nav_state}", flush=True)
        if isinstance(data, str):
            if "[Localization] initialization failed!" in data:
                self.state.localization_init_state = "failed"
                self.resolve_localization_init_waiters()
            elif "[Localization] initialization succeed!" in data:
                self.state.localization_init_state = "succeeded"
                self.resolve_localization_init_waiters()
            if data.startswith("localization/get_status/status/"):
                self.resolve_slam_response_waiters(
                    "localization/get_status",
                    data.rsplit("/", 1)[-1].strip(),
                )
            elif data.startswith("navigation/get_status/status/"):
                self.resolve_slam_response_waiters(
                    "navigation/get_status",
                    data.rsplit("/", 1)[-1].strip(),
                )
            elif data.startswith("patrol/get_status/status/"):
                self.resolve_slam_response_waiters(
                    "patrol/get_status",
                    data.rsplit("/", 1)[-1].strip(),
                )
            elif data.startswith("mapping/get_status/status/"):
                self.resolve_slam_response_waiters(
                    "mapping/get_status",
                    data.rsplit("/", 1)[-1].strip(),
                )
            elif data.startswith("common/get_map_id/map_id/"):
                self.resolve_slam_response_waiters(
                    "common/get_map_id",
                    data[len("common/get_map_id/map_id/"):].strip(),
                )
        if config.SLAM_DEBUG_LOGS:
            print(f"[SLAM Server] {data}", flush=True)


    def build_door_approach_sequence(self, door_name: str) -> list[dict]:
        return door_approach_sequence(door_name, self.map_service.match_location_exact)


    def build_door_post_permission_sequence(self, door_name: str) -> list[dict]:
        return []


    def build_corridor_transition_sequence(self, destination_pose: dict) -> list[dict]:
        chain: list[dict] = []
        for location_name in config.CORRIDOR_APPROACH_CHAIN:
            pose = self.map_service.match_location_exact(location_name)
            if pose is not None:
                chain.append(pose)

        if len(chain) < 2:
            return []

        destination_name = str(destination_pose.get("name", destination_pose.get("kind", "destination"))).strip()
        destination_key = normalize_name_key(destination_name)
        chain_keys = {
            normalize_name_key(str(pose.get("name", pose.get("kind", ""))).strip())
            for pose in chain
        }
        if destination_key in chain_keys:
            return []

        current_y = float(self.state.current_pose["y"])
        destination_y = float(destination_pose["y"])
        epsilon = 0.15

        if destination_y > current_y + epsilon:
            selected = [
                pose for pose in chain
                if (current_y + epsilon) < float(pose["y"]) < (destination_y - epsilon)
            ]
        elif destination_y < current_y - epsilon:
            selected = [
                with_rotated_yaw(pose, math.pi) for pose in reversed(chain)
                if (destination_y + epsilon) < float(pose["y"]) < (current_y - epsilon)
            ]
        else:
            selected = []

        if not selected:
            return []

        selected_names = [pose.get("name", pose.get("kind", "?")) for pose in selected]
        print(
            f"[CorridorRoute] Inserting corridor route before {destination_name}: "
            f"{' -> '.join(selected_names)}",
            flush=True,
        )
        return selected


    async def reverse_from_door_to_waypoint(self, door_pose: dict, waypoint_pose: dict) -> dict:
        try:
            dx = float(waypoint_pose["x"]) - float(door_pose["x"])
            dy = float(waypoint_pose["y"]) - float(door_pose["y"])
            distance = (dx * dx + dy * dy) ** 0.5
        except Exception as exc:
            return {"status": "error", "message": f"Invalid reverse geometry: {exc}"}

        if distance <= 0.02:
            return {
                "status": "ok",
                "location": waypoint_pose.get("name", waypoint_pose.get("kind", "door_waypoint")),
                "distance_m": distance,
                "mode": "reverse_skip",
            }

        duration_sec = config.DOOR_REVERSE_DURATION_SEC
        distance = min(distance, config.DOOR_REVERSE_MAX_DISTANCE_M)
        pose_before = dict(self.state.current_pose)
        yaw_before = float(pose_before.get("yaw", 0.0))

        try:
            self.send_slam_cmd("navigation/stop")
            await asyncio.sleep(config.DOOR_REVERSE_PREPARE_SEC)
            self.send_slam_cmd("common/enable_joystick_control")
            await asyncio.sleep(0.3)
            elapsed = 0.0
            while elapsed < duration_sec:
                self.publish_wireless_controller(ly=config.DOOR_REVERSE_WIRELESS_LY)
                await asyncio.sleep(config.DOOR_REVERSE_COMMAND_INTERVAL_SEC)
                elapsed += config.DOOR_REVERSE_COMMAND_INTERVAL_SEC
            self.publish_wireless_controller()
            await asyncio.sleep(0.2)
            self.send_slam_cmd("common/disable_joystick_control")
            await asyncio.sleep(config.DOOR_REVERSE_SETTLE_SEC)
            pose_after = dict(self.state.current_pose)
            dx_world = float(pose_after.get("x", 0.0)) - float(pose_before.get("x", 0.0))
            dy_world = float(pose_after.get("y", 0.0)) - float(pose_before.get("y", 0.0))
            dx_body = math.cos(yaw_before) * dx_world + math.sin(yaw_before) * dy_world
            dy_body = -math.sin(yaw_before) * dx_world + math.cos(yaw_before) * dy_world
            print(
                "[Door] Reverse pose delta "
                f"world=({dx_world:+.3f}, {dy_world:+.3f}) "
                f"body=({dx_body:+.3f}, {dy_body:+.3f}) "
                f"yaw_before={yaw_before:+.3f}",
                flush=True,
            )
            return {
                "status": "ok",
                "location": waypoint_pose.get("name", waypoint_pose.get("kind", "door_waypoint")),
                "distance_m": distance,
                "duration_sec": duration_sec,
                "mode": "reverse_wireless_controller",
                "pose_before": pose_before,
                "pose_after": pose_after,
                "body_dx_m": dx_body,
                "body_dy_m": dy_body,
            }
        except Exception as exc:
            try:
                self.publish_wireless_controller()
            except Exception:
                pass
            try:
                self.send_slam_cmd("common/disable_joystick_control")
            except Exception:
                pass
            return {"status": "error", "message": f"Reverse move failed: {exc}"}


    async def reverse_from_current_door_if_needed(self, target_name: str) -> Optional[dict]:
        nearest_pose, nearest_distance = self.get_nearest_named_location()
        if nearest_pose is None or nearest_distance > config.DOOR_REVERSE_TRIGGER_DISTANCE_M:
            return None

        current_name = str(nearest_pose.get("name", "")).strip()
        current_key = normalize_name_key(current_name)
        if not current_key.startswith("door") or "waypoint" in current_key:
            return None

        target_key = normalize_name_key(target_name)
        if target_key == current_key:
            return None

        reverse_waypoint = (
            self.map_service.match_location_exact(f"{current_name}_waypoint1")
            or self.map_service.match_location_exact(f"{current_name}_waypoint")
        )
        if reverse_waypoint is None:
            print(
                f"[Door] No reverse waypoint configured for {current_name}; starting navigation in place",
                flush=True,
            )
            return None

        print(
            f"[Door] Current position is near {current_name} "
            f"(dist={nearest_distance:.3f}); reversing toward "
            f"{reverse_waypoint.get('name', reverse_waypoint.get('kind', 'door_waypoint'))} before navigation",
            flush=True,
        )
        return await self.reverse_from_door_to_waypoint(nearest_pose, reverse_waypoint)


    def send_slam_cmd(self, cmd: str):
        self.state.conn.datachannel.pub_sub.publish_without_callback(RTC_TOPIC["LIDAR_MAPPING_CMD"], cmd)
        print(f"[SLAM] {cmd}", flush=True)


    def publish_wireless_controller(self, 
        *,
        lx: float = 0.0,
        ly: float = 0.0,
        rx: float = 0.0,
        ry: float = 0.0,
        keys: int = 0,
    ) -> None:
        payload = {"lx": lx, "ly": ly, "rx": rx, "ry": ry, "keys": keys}
        self.state.conn.datachannel.pub_sub.publish_without_callback(RTC_TOPIC["WIRELESS_CONTROLLER"], payload)
        print(
            f"[Wireless] lx={lx:+.3f} ly={ly:+.3f} rx={rx:+.3f} ry={ry:+.3f} keys={keys}",
            flush=True,
        )


    async def query_slam_response(self, command_key: str, timeout_sec: float = 2.0) -> str:
        if self.state.no_robot or self.state.conn is None:
            return ""

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        waiters = self.state.slam_response_waiters.setdefault(command_key, [])
        waiters.append((loop, future))
        self.send_slam_cmd(command_key)

        try:
            return await asyncio.wait_for(future, timeout=timeout_sec)
        except asyncio.TimeoutError:
            return ""
        finally:
            remaining = [
                (wait_loop, wait_future)
                for wait_loop, wait_future in self.state.slam_response_waiters.get(command_key, [])
                if wait_future is not future
            ]
            if remaining:
                self.state.slam_response_waiters[command_key] = remaining
            else:
                self.state.slam_response_waiters.pop(command_key, None)


    async def wait_for_localization_init_state(self, timeout_sec: float) -> Optional[str]:
        if self.state.localization_init_state in {"failed", "succeeded"}:
            return self.state.localization_init_state

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        waiter = (loop, future)
        self.state.localization_init_waiters.append(waiter)
        try:
            return await asyncio.wait_for(future, timeout=timeout_sec)
        except asyncio.TimeoutError:
            return None
        finally:
            if waiter in self.state.localization_init_waiters:
                self.state.localization_init_waiters.remove(waiter)


    async def probe_slam_startup_state(self, ) -> dict[str, str]:
        async def safe_query(command_key: str) -> str:
            try:
                return await self.query_slam_response(command_key)
            except Exception as e:
                print(f"[SLAM] Failed to query {command_key}: {e}", flush=True)
                return ""

        mapping, localization, patrol, navigation, map_id = await asyncio.gather(
            safe_query("mapping/get_status"),
            safe_query("localization/get_status"),
            safe_query("patrol/get_status"),
            safe_query("navigation/get_status"),
            safe_query("common/get_map_id"),
        )
        print(
            "[SLAM] Startup probe "
            f"mapping={mapping or '?'} "
            f"localization={localization or '?'} "
            f"patrol={patrol or '?'} "
            f"navigation={navigation or '?'} "
            f"map_id={map_id or '?'}",
            flush=True,
        )
        return {
            "mapping": mapping,
            "localization": localization,
            "patrol": patrol,
            "navigation": navigation,
            "map_id": map_id,
        }


    async def stop_active_slam_modules_on_shutdown(self, ) -> None:
        if self.state.no_robot:
            return
        channel = getattr(getattr(self.state.conn, "datachannel", None), "channel", None)
        if self.state.conn is None or channel is None:
            print("[Shutdown] Skipping SLAM stop commands because there is no data channel", flush=True)
            return
        if not self.connection.is_ready():
            print(
                "[Shutdown] WebRTC connection looks degraded; attempting SLAM stop commands anyway "
                "so the robot doesn't keep a stale localization/navigation session running",
                flush=True,
            )

        print("[Shutdown] Stopping navigation/patrol before disconnect and keeping localization alive", flush=True)
        shutdown_cmds = [
            "navigation/stop",
            "patrol/stop",
        ]
        for cmd in shutdown_cmds:
            try:
                self.send_slam_cmd(cmd)
            except Exception as e:
                print(f"[Shutdown] Failed to send {cmd}: {e}", flush=True)
            await asyncio.sleep(0.25)


    async def activate_office210_map_once(self, ):
        if self.state.no_robot:
            return
        if not self.state.active_map_id:
            return
        self.send_slam_cmd(f"common/set_map_id/{self.state.active_map_id}")
        await asyncio.sleep(0.8)


    async def reset_stale_slam_session(self, ):
        if self.state.no_robot:
            return
        print("[SLAM] Clearing any stale localization/navigation session before start", flush=True)
        self.send_slam_cmd("patrol/stop")
        await asyncio.sleep(0.3)
        self.send_slam_cmd("navigation/stop")
        await asyncio.sleep(0.3)
        self.send_slam_cmd("localization/stop")
        await asyncio.sleep(1.5)


    def localization_pose_looks_valid(self, before_pose: dict, after_pose: dict, before_update_ts: float) -> bool:
        moved = (
            abs(after_pose["x"] - before_pose["x"])
            + abs(after_pose["y"] - before_pose["y"])
        )
        has_fresh_update = self.state.last_pose_update_monotonic > before_update_ts
        not_zero_pose = any(abs(after_pose[key]) > 1e-6 for key in ("x", "y", "yaw"))
        return has_fresh_update and (moved > 1e-4 or not_zero_pose)


    async def start_localization_attempt(self, attempt_index: int, max_attempts: int) -> bool:
        print(
            f"[SLAM] Localization start attempt {attempt_index}/{max_attempts}",
            flush=True,
        )
        self.state.localization_init_state = "pending"
        if self.state.initial_pose:
            print(
                f"[SLAM] Setting initial pose x={self.state.initial_pose['x']:.3f} "
                f"y={self.state.initial_pose['y']:.3f} yaw={self.state.initial_pose['yaw']:.3f}",
                flush=True,
            )
            self.send_slam_cmd(
                "localization/set_initial_pose/"
                f"{self.state.initial_pose['x']:.3f}/{self.state.initial_pose['y']:.3f}/{self.state.initial_pose['yaw']:.3f}"
            )
            await asyncio.sleep(0.5)

        pose_before = dict(self.state.current_pose)
        before_update_ts = self.state.last_pose_update_monotonic
        self.send_slam_cmd("localization/start")
        await asyncio.sleep(0.5)
        print("[SLAM] Localization started", flush=True)

        init_state_task = asyncio.create_task(
            self.wait_for_localization_init_state(config.LOCALIZATION_INIT_STATE_TIMEOUT_SEC)
        )
        await asyncio.sleep(config.LOCALIZATION_POSE_SETTLE_SEC)
        pose_after = dict(self.state.current_pose)
        moved = (
            abs(pose_after["x"] - pose_before["x"])
            + abs(pose_after["y"] - pose_before["y"])
        )
        print(
            f"[SLAM] Pose {config.LOCALIZATION_POSE_SETTLE_SEC:.1f}s after localization/start: "
            f"x={pose_after['x']:.3f} y={pose_after['y']:.3f} yaw={pose_after['yaw']:.3f} "
            f"(drift_since_start={moved:.3f})",
            flush=True,
        )

        init_state = await init_state_task
        pose_valid = self.localization_pose_looks_valid(pose_before, pose_after, before_update_ts)
        if init_state == "succeeded":
            print("[SLAM] Localization initialization succeeded", flush=True)
            return True
        if init_state == "failed":
            print("[SLAM] Localization initialization failed; will retry if attempts remain", flush=True)
            return False
        if pose_valid:
            print("[SLAM] Localization pose updates look healthy even without explicit success log", flush=True)
            return True

        print(
            "[SLAM] WARNING: no pose update received after localization/start - "
            "localization may not have converged (possible stale/duplicate session on the robot, "
            "or point cloud not matching against the loaded map).",
            flush=True,
        )
        return False


    async def start_localization_once(self, ):
        if self.state.no_robot:
            return
        startup_state = await self.probe_slam_startup_state()
        robot_map_id = startup_state.get("map_id", "").strip()
        if robot_map_id:
            print(f"[SLAM] Robot reports active map id: {robot_map_id}", flush=True)
            if not self.state.active_map_id:
                self.state.active_map_id = robot_map_id

        if startup_state.get("localization") == "1":
            print("[SLAM] Existing localization is already active; reusing robot state", flush=True)
            if startup_state.get("patrol") == "1":
                print("[SLAM] Stopping active patrol while keeping localization alive", flush=True)
                self.send_slam_cmd("patrol/stop")
                await asyncio.sleep(0.5)
            if startup_state.get("navigation") == "1":
                print("[SLAM] Stopping active navigation while keeping localization alive", flush=True)
                self.send_slam_cmd("navigation/stop")
                await asyncio.sleep(0.5)
            return

        if startup_state.get("mapping") == "1":
            print("[SLAM] Mapping is active on the robot; stopping it before localization startup", flush=True)
            self.send_slam_cmd("mapping/stop")
            await asyncio.sleep(0.8)

        await self.reset_stale_slam_session()
        print(f"[SLAM] Requesting map activation, map_id={self.state.active_map_id!r}", flush=True)
        await self.activate_office210_map_once()
        for attempt_index in range(1, config.LOCALIZATION_INIT_MAX_ATTEMPTS + 1):
            success = await self.start_localization_attempt(attempt_index, config.LOCALIZATION_INIT_MAX_ATTEMPTS)
            if success:
                return
            if attempt_index >= config.LOCALIZATION_INIT_MAX_ATTEMPTS:
                print(
                    "[SLAM] Localization failed after automatic retries. "
                    "Check map alignment / point cloud / frontend state before starting navigation.",
                    flush=True,
                )
                return
            print("[SLAM] Resetting localization before retry", flush=True)
            self.send_slam_cmd("localization/stop")
            await asyncio.sleep(1.5)
            await self.activate_office210_map_once()


    def log_current_pose_and_nearest_waypoint(self, ):
        print(
            f"[SLAM] Pose x={self.state.current_pose['x']:.3f} y={self.state.current_pose['y']:.3f} "
            f"yaw={self.state.current_pose['yaw']:.3f}",
            flush=True,
        )
        if not self.state.named_locations:
            return

        nearest_index = -1
        nearest_distance = float("inf")
        locations = list(self.state.named_locations.values())
        for i, wp in enumerate(locations):
            dx = self.state.current_pose["x"] - wp["x"]
            dy = self.state.current_pose["y"] - wp["y"]
            distance = (dx * dx + dy * dy) ** 0.5
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_index = i

        if nearest_index >= 0:
            print(
                f"[SLAM] Nearest location: {locations[nearest_index]['name']} "
                f"(dist={nearest_distance:.3f})",
                flush=True,
            )


    def get_nearest_named_location(self, ) -> tuple[Optional[dict], float]:
        if not self.state.named_locations:
            self.map_service.load_named_locations()
        if not self.state.named_locations:
            return None, float("inf")

        nearest_pose = None
        nearest_distance = float("inf")
        for pose in self.state.named_locations.values():
            dx = self.state.current_pose["x"] - pose["x"]
            dy = self.state.current_pose["y"] - pose["y"]
            distance = (dx * dx + dy * dy) ** 0.5
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_pose = pose
        return nearest_pose, nearest_distance


    def capture_return_target(self, ) -> Optional[dict]:
        nearest_pose, nearest_distance = self.get_nearest_named_location()
        if nearest_pose is not None and nearest_distance <= config.RETURN_TO_NAMED_LOCATION_MAX_DISTANCE:
            return {
                "type": "named_location",
                "location": nearest_pose["name"],
                "distance": nearest_distance,
            }

        pose_age = time.monotonic() - self.state.last_pose_update_monotonic if self.state.last_pose_update_monotonic else float("inf")
        if pose_age <= config.RETURN_TO_POSE_MAX_AGE_SEC:
            return {
                "type": "pose",
                "location": "starting position",
                "x": self.state.current_pose["x"],
                "y": self.state.current_pose["y"],
                "yaw": self.state.current_pose["yaw"],
                "pose_age_sec": pose_age,
            }

        if nearest_pose is not None:
            return {
                "type": "named_location",
                "location": nearest_pose["name"],
                "distance": nearest_distance,
            }
        return None


    async def navigate_to_pose(self, wp: dict) -> dict:
        target_name = wp.get("name", "target")
        target_kind = str(wp.get("kind", target_name)).strip().lower()
        last_error = None
        
        def distance_to_target() -> float:
            try:
                dx = float(self.state.current_pose["x"]) - float(wp["x"])
                dy = float(self.state.current_pose["y"]) - float(wp["y"])
                return (dx * dx + dy * dy) ** 0.5
            except Exception:
                return float("inf")

        def point_one_close_enough_result(mode: str) -> Optional[dict]:
            if target_kind != "point one":
                return None
            if mode not in {"no_path", "failure"}:
                return None
            current_distance = distance_to_target()
            if current_distance > config.POINT_ONE_CLOSE_ENOUGH_THRESHOLD_M:
                return None
            print(
                f"[SLAM] Treating {target_name} as reached after {mode} "
                f"(dist={current_distance:.3f} <= {config.POINT_ONE_CLOSE_ENOUGH_THRESHOLD_M:.2f}m)",
                flush=True,
            )
            return {
                "status": "ok",
                "location": target_name,
                "x": self.state.current_pose["x"],
                "y": self.state.current_pose["y"],
                "yaw": self.state.current_pose["yaw"],
                "distance_m": current_distance,
                "mode": f"close_enough_after_{mode}",
            }

        current_distance = distance_to_target()

        if current_distance <= config.NAVIGATION_ALREADY_AT_TARGET_DISTANCE_M:
            print(
                f"[SLAM] Already near {target_name} "
                f"(dist={current_distance:.3f}), skipping navigation",
                flush=True,
            )
            return {
                "status": "ok",
                "location": target_name,
                "x": self.state.current_pose["x"],
                "y": self.state.current_pose["y"],
                "yaw": self.state.current_pose["yaw"],
                "distance_m": current_distance,
                "mode": "already_there",
            }

        for attempt in range(1, config.NAVIGATION_MAX_ATTEMPTS + 1):
            self.log_current_pose_and_nearest_waypoint()
            if attempt > 1:
                print(
                    f"[SLAM] Retrying navigation to {target_name} "
                    f"({attempt}/{config.NAVIGATION_MAX_ATTEMPTS})",
                    flush=True,
                )
                self.send_slam_cmd("navigation/stop")
                await asyncio.sleep(config.NAVIGATION_RETRY_DELAY_SEC)

            self.send_slam_cmd("navigation/start")
            await asyncio.sleep(0.5)
            self.state.nav_state = None
            start_seq = self.state.nav_state_seq
            self.send_slam_cmd(f"navigation/set_goal_pose/{wp['x']}/{wp['y']}/{wp.get('yaw', 0.0)}")
            print(f"[SLAM] Moving to {target_name}", flush=True)

            deadline = time.monotonic() + config.LOCATION_TIMEOUT_SEC
            first_state_deadline = time.monotonic() + config.NAVIGATION_FIRST_STATE_TIMEOUT_SEC
            while time.monotonic() < deadline:
                if self.state.stop_event.is_set():
                    return {
                        "status": "error",
                        "message": f"Navigation to {target_name} was cancelled.",
                    }
                if self.state.nav_state_seq > start_seq:
                    if self.state.nav_state == "REACHED":
                        print(f"[SLAM] Arrived at {target_name}", flush=True)
                        return {
                            "status": "ok",
                            "location": target_name,
                            "x": self.state.current_pose["x"],
                            "y": self.state.current_pose["y"],
                            "yaw": self.state.current_pose["yaw"],
                        }
                    if self.state.nav_state in config.NAV_FAILURE_STATES:
                        last_error = f"Navigation to {target_name} failed: {self.state.nav_state}"
                        print(f"[SLAM] {last_error}", flush=True)
                        close_enough_result = point_one_close_enough_result(self.state.nav_state.lower())
                        if close_enough_result is not None:
                            return close_enough_result
                        break
                elif time.monotonic() >= first_state_deadline:
                    navigation_status = await self.query_slam_response("navigation/get_status")
                    localization_status = await self.query_slam_response("localization/get_status")
                    last_error = (
                        f"No navigation state transition observed within "
                        f"{config.NAVIGATION_FIRST_STATE_TIMEOUT_SEC:.1f}s while moving to {target_name} "
                        f"(navigation_status={navigation_status or '?'}, "
                        f"localization_status={localization_status or '?'})"
                    )
                    print(f"[SLAM] {last_error}", flush=True)
                    break
                await asyncio.sleep(0.2)
            else:
                last_error = f"Timed out while moving to {target_name}."
                print(f"[SLAM] {last_error}", flush=True)

        return {
            "status": "error",
            "message": last_error or f"Navigation to {target_name} failed.",
            "attempts": config.NAVIGATION_MAX_ATTEMPTS,
        }


    async def navigate_to_waypoint_index(self, index: int) -> dict:
        locations = list(self.state.named_locations.values())
        if index < 0 or index >= len(locations):
            return {"status": "error", "message": f"Waypoint {index + 1} does not exist."}
        return await self.navigate_to_pose(locations[index])

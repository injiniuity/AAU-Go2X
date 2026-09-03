import asyncio
from collections.abc import Awaitable, Callable
from typing import Any


class RobotConnectionService:
    def __init__(
        self,
        *,
        state: Any,
        connection_cls: type,
        connection_method: Any,
        robot_ip: str,
        initial_retries: int,
        retry_delay_sec: float,
        build_audio_track: Callable[[], Any],
        register_streams_and_callbacks: Callable[[Any], None],
        load_active_map_id: Callable[[], None],
        load_named_locations: Callable[[], None],
        start_localization_once: Callable[[], Awaitable[None]],
    ) -> None:
        self._state = state
        self._connection_cls = connection_cls
        self._connection_method = connection_method
        self._robot_ip = robot_ip
        self._initial_retries = initial_retries
        self._retry_delay_sec = retry_delay_sec
        self._build_audio_track = build_audio_track
        self._register_streams_and_callbacks = register_streams_and_callbacks
        self._load_active_map_id = load_active_map_id
        self._load_named_locations = load_named_locations
        self._start_localization_once = start_localization_once

    def is_ready(self) -> bool:
        if self._state.no_robot or self._state.conn is None:
            return False
        pc = getattr(self._state.conn, "pc", None)
        channel = getattr(getattr(self._state.conn, "datachannel", None), "channel", None)
        return bool(
            self._state.conn
            and getattr(self._state.conn, "isConnected", False)
            and pc
            and getattr(pc, "connectionState", "") == "connected"
            and channel
            and getattr(channel, "readyState", "") == "open"
        )

    def outgoing_audio_track_ready(self) -> bool:
        return bool(
            self._state.audio_track is not None
            and getattr(self._state.audio_track, "_ready", False)
        )

    async def ensure_outgoing_audio_track(self, force_rebuild: bool = False) -> bool:
        if self._state.text_mode or self._state.no_robot:
            return True
        if self._state.conn is None:
            return False
        if not force_rebuild and self.outgoing_audio_track_ready():
            return True
        try:
            self._state.audio_track = self._build_audio_track()
            self._state.conn.attach_outgoing_audio_track(self._state.audio_track)
            await asyncio.sleep(0.5)
            self._state.audio_track.set_ready()
            print("[Audio] Outgoing audio track attached", flush=True)
            return True
        except Exception as exc:
            print(f"[Audio] Failed to attach outgoing audio track: {exc}", flush=True)
            return False

    async def refresh_outgoing_audio_for_playback(self) -> bool:
        if self._state.text_mode or self._state.no_robot:
            return True
        if self._state.conn is None:
            return False
        try:
            self._state.conn.audio.switchAudioChannel(True)
            print("[Audio] Audio channel re-enabled before playback", flush=True)
        except Exception as exc:
            print(f"[Audio] Failed to re-enable audio channel before playback: {exc}", flush=True)
            return False
        return await self.ensure_outgoing_audio_track(force_rebuild=True)

    async def ensure_robot_connection(self) -> bool:
        if self._state.no_robot:
            return True
        if self.is_ready():
            if not self._state.text_mode and not self.outgoing_audio_track_ready():
                await self.ensure_outgoing_audio_track(force_rebuild=True)
            return True

        print("[WebRTC] Connection lost. Reconnecting...")
        try:
            await self._state.conn.connect()
            if not self._state.text_mode:
                self._state.conn.audio.switchAudioChannel(True)
                await self.ensure_outgoing_audio_track(force_rebuild=True)
            self._register_streams_and_callbacks(self._state.conn)
            self._load_active_map_id()
            self._load_named_locations()
            await self._start_localization_once()
            print("[WebRTC] Reconnected")
            return True
        except Exception as exc:
            print(f"[WebRTC] Reconnect failed: {exc}")
            return False

    async def connect_with_retries(self):
        if not self._robot_ip:
            raise RuntimeError(
                "UNITREE_ROBOT_IP is not configured. Set it in .env before running with a robot."
            )

        last_error = None

        for attempt in range(1, self._initial_retries + 1):
            current_conn = self._connection_cls(self._connection_method.LocalSTA, ip=self._robot_ip)
            try:
                print(
                    f"[WebRTC] Initial connect attempt {attempt}/{self._initial_retries}",
                    flush=True,
                )
                await current_conn.connect()
                return current_conn
            except Exception as exc:
                last_error = exc
                print(f"[WebRTC] Initial connect attempt {attempt} failed: {exc}", flush=True)
                try:
                    await current_conn.disconnect()
                except Exception:
                    pass

                if attempt < self._initial_retries:
                    print(
                        f"[WebRTC] Waiting {self._retry_delay_sec:.0f}s before retry...",
                        flush=True,
                    )
                    await asyncio.sleep(self._retry_delay_sec)

        raise RuntimeError(
            f"Failed to connect to the robot after {self._initial_retries} attempts. "
            f"Last error: {last_error}"
        )

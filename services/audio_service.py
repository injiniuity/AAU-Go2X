import asyncio
import base64
import io
import json
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .. import config
from ..utils.audio_utils import (
    collect_audiohub_records,
    get_audiohub_record_create_time,
    get_audiohub_record_name,
    get_audiohub_record_uuid,
    get_wav_duration_sec,
    prepend_silence_to_wav_bytes,
    sanitize_audiohub_record_name,
    select_audiohub_record_by_name,
)
from ..utils.logging_utils import print
from ..utils.text_utils import normalize_name_key, sanitize_for_tts

from unitree_webrtc_connect.webrtc_audiohub import WebRTCAudioHub


class AudioService:
    """Owns AudioHub playback, generated TTS, and the outgoing WebRTC track."""

    def __init__(self, *, state, connection, client) -> None:
        self.state = state
        self.connection = connection
        self.client = client

    def resolve_audiohub_play_waiters(self) -> None:
        remaining = []
        snapshot = dict(self.state.audiohub_play_state)
        for loop, future, predicate in self.state.audiohub_play_waiters:
            if future.done():
                continue
            try:
                matched = bool(predicate(snapshot))
            except Exception:
                matched = False
            if matched:
                loop.call_soon_threadsafe(future.set_result, snapshot)
            else:
                remaining.append((loop, future, predicate))
        self.state.audiohub_play_waiters = remaining


    def audiohub_play_state_callback(self, msg):
        try:
            data = msg.get("data", msg)
            if isinstance(data, str):
                data = json.loads(data)
            if not isinstance(data, dict):
                return
            self.state.audiohub_play_state = {
                "is_playing": bool(data.get("is_playing", False)),
                "current_audio_unique_id": str(data.get("current_audio_unique_id", "")).strip(),
                "current_audio_custom_name": str(data.get("current_audio_custom_name", "")).strip(),
            }
            self.resolve_audiohub_play_waiters()
        except Exception as e:
            print(f"[AudioHub State Error] {e}", flush=True)


    async def wait_for_audiohub_play_state(self, predicate, timeout_sec: float) -> Optional[dict]:
        snapshot = dict(self.state.audiohub_play_state)
        try:
            if predicate(snapshot):
                return snapshot
        except Exception:
            pass

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        waiter = (loop, future, predicate)
        self.state.audiohub_play_waiters.append(waiter)
        try:
            return await asyncio.wait_for(future, timeout=timeout_sec)
        except asyncio.TimeoutError:
            return None
        finally:
            if waiter in self.state.audiohub_play_waiters:
                self.state.audiohub_play_waiters.remove(waiter)


    def get_audio_hub(self) -> WebRTCAudioHub:
        if self.state.audio_hub is None or getattr(self.state.audio_hub, "conn", None) is not self.state.conn:
            self.state.audio_hub = WebRTCAudioHub(self.state.conn)
        return self.state.audio_hub


    async def _play_audiohub_uuid(self, record_uuid: str, duration_sec: float, label: str) -> bool:
        print(f"[AudioHub] {label}")
        await self.get_audio_hub().play_by_uuid(record_uuid)
        print(f"[AudioHub] Waiting for uuid {record_uuid} playback state...")
        started_state = await self.wait_for_audiohub_play_state(
            lambda play_state: (
                play_state.get("is_playing") is True
                and play_state.get("current_audio_unique_id") == record_uuid
            ),
            timeout_sec=max(2.0, min(duration_sec, 6.0)),
        )
        if started_state is None:
            print(
                f"[AudioHub] Start state not observed for {record_uuid}; "
                f"falling back to duration wait ({duration_sec:.1f}s)",
                flush=True,
            )
            await asyncio.sleep(duration_sec)
        else:
            print(
                f"[AudioHub] Started uuid {record_uuid}"
                + (
                    f" ({started_state.get('current_audio_custom_name')})"
                    if started_state.get("current_audio_custom_name")
                    else ""
                ),
                flush=True,
            )
            stopped_state = await self.wait_for_audiohub_play_state(
                lambda play_state: (
                play_state.get("is_playing") is False
                and play_state.get("current_audio_unique_id") in {"", record_uuid}
                ),
                timeout_sec=max(duration_sec + 3.0, 6.0),
            )
            if stopped_state is None:
                print(
                    f"[AudioHub] Stop state not observed for {record_uuid}; "
                    f"falling back to tail wait ({duration_sec:.1f}s)",
                    flush=True,
                )
                await asyncio.sleep(duration_sec)
        await asyncio.sleep(config.AUDIO_PLAYBACK_TAIL_SEC)
        print("[AudioHub] Done")
        return True


    async def play_audiohub_record(self, record_uuid: str, duration_sec: float, label: str) -> bool:
        async with self.state.speak_lock:
            self.state.tts_active.set()
            if self.state.no_robot:
                print("[AudioHub] Skipped in --no-robot mode")
                self.state.tts_active.clear()
                return False
            if not await self.connection.ensure_robot_connection():
                print("[AudioHub] Skipped because robot connection is down")
                self.state.tts_active.clear()
                return False
            try:
                return await self._play_audiohub_uuid(record_uuid, duration_sec, label)
            finally:
                self.state.tts_active.clear()


    async def play_audiohub_uploaded_wav(
        wav_bytes: bytes,
        label: str,
        prefix_silence_sec: float = 0.0,
    ) -> bool:
        wav_bytes = prepend_silence_to_wav_bytes(wav_bytes, prefix_silence_sec)
        duration_sec = get_wav_duration_sec(wav_bytes)
        config.AUDIOHUB_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        unique_stem = (
            "tts_"
            + time.strftime("%Y%m%d_%H%M%S")
            + "_"
            + str(time.monotonic_ns())[-8:]
            + "_"
            + sanitize_audiohub_record_name(label)
        )
        temp_wav_path = config.AUDIOHUB_UPLOAD_DIR / f"{unique_stem}.wav"
        temp_wav_path.write_bytes(wav_bytes)

        hub = self.get_audio_hub()
        record_uuid = ""
        after_records: list[dict[str, Any]] = []
        try:
            before_response = await hub.get_audio_list()
            before_records = collect_audiohub_records(before_response)
            before_ids = {
                get_audiohub_record_uuid(record)
                for record in before_records
                if get_audiohub_record_uuid(record)
            }
            before_first_uuid = get_audiohub_record_uuid(before_records[0]) if before_records else ""
            before_count = len(before_records)

            print(f"[AudioHub] Uploading generated TTS as {temp_wav_path.name}", flush=True)
            await hub.upload_audio_file(str(temp_wav_path))

            matched_record = None
            for _ in range(config.AUDIOHUB_UPLOAD_POLL_ATTEMPTS):
                await asyncio.sleep(config.AUDIOHUB_UPLOAD_POLL_DELAY_SEC)
                after_response = await hub.get_audio_list()
                after_records = collect_audiohub_records(after_response)
                if after_records:
                    first_record = after_records[0]
                    first_uuid = get_audiohub_record_uuid(first_record)
                    first_name = get_audiohub_record_name(first_record)
                    if (
                        first_uuid
                        and (
                            len(after_records) > before_count
                            or first_uuid != before_first_uuid
                            or normalize_name_key(first_name) == normalize_name_key(unique_stem)
                        )
                    ):
                        print(
                            "[AudioHub] Using first audio-list entry for uploaded TTS",
                            flush=True,
                        )
                        matched_record = first_record
                if matched_record is None:
                    for record in after_records:
                        candidate_uuid = get_audiohub_record_uuid(record)
                        if candidate_uuid and candidate_uuid not in before_ids:
                            matched_record = record
                            break
                if matched_record is None:
                    matched_record = select_audiohub_record_by_name(after_records, unique_stem)
                if matched_record is not None:
                    break

            if matched_record is None:
                preview = ", ".join(
                    f"{get_audiohub_record_name(record) or '<unnamed>'} ({get_audiohub_record_uuid(record) or 'no-uuid'})"
                    for record in after_records[:3]
                ) or "<empty>"
                print(
                    "[AudioHub] Uploaded TTS record UUID could not be identified. "
                    f"First entries: {preview}",
                    flush=True,
                )
                return False

            record_uuid = get_audiohub_record_uuid(matched_record)
            if not record_uuid:
                print("[AudioHub] Uploaded TTS record had no UUID", flush=True)
                return False

            return await self._play_audiohub_uuid(record_uuid, duration_sec, label)
        finally:
            try:
                if record_uuid:
                    await hub.delete_record(record_uuid)
            except Exception as delete_error:
                print(f"[AudioHub] Failed to delete uploaded TTS record {record_uuid}: {delete_error}", flush=True)
            try:
                temp_wav_path.unlink(missing_ok=True)
            except Exception as unlink_error:
                print(f"[AudioHub] Failed to remove temp WAV {temp_wav_path}: {unlink_error}", flush=True)


    async def play_prompt_audio(
        *,
        label: str,
        wav_path: Optional[Path] = None,
        fallback_uuid: Optional[str] = None,
        fallback_duration_sec: float = 0.0,
    ) -> bool:
        if fallback_uuid:
            return await self.play_audiohub_record(fallback_uuid, fallback_duration_sec, label=label)
        if wav_path is not None and wav_path.exists():
            return await self.play_wav_file(wav_path, label=label)
        print(f"[Audio] No playable source available for: {label}", flush=True)
        return False


    def build_audio_track(self):
        import av
        from aiortc.mediastreams import AudioStreamTrack
        from fractions import Fraction
        from pydub import AudioSegment

        class QueuedAudioTrack(AudioStreamTrack):
            kind = "audio"

            def __init__(self):
                super().__init__()
                self._queue: asyncio.Queue = asyncio.Queue()
                self._pts = 0
                self._ready = False

            def set_ready(self):
                self._ready = True

            def _silence_chunk(self) -> np.ndarray:
                return np.zeros((config.ROBOT_SAMPLES_PER_FRAME, config.ROBOT_AUDIO_CHANNELS), dtype=np.int16)

            def _build_frame(self, pcm: np.ndarray) -> av.AudioFrame:
                packed = pcm.reshape(1, -1)
                layout = "mono" if config.ROBOT_AUDIO_CHANNELS == 1 else "stereo"
                frame = av.AudioFrame.from_ndarray(packed, format="s16", layout=layout)
                frame.sample_rate = config.ROBOT_AUDIO_SAMPLE_RATE
                frame.pts = self._pts
                frame.time_base = Fraction(1, config.ROBOT_AUDIO_SAMPLE_RATE)
                self._pts += config.ROBOT_SAMPLES_PER_FRAME
                return frame

            async def recv(self):
                if not self._ready:
                    await asyncio.sleep(config.ROBOT_SAMPLES_PER_FRAME / config.ROBOT_AUDIO_SAMPLE_RATE)
                    return self._build_frame(self._silence_chunk())
                try:
                    pcm, playback_done = await asyncio.wait_for(
                        self._queue.get(),
                        timeout=config.ROBOT_SAMPLES_PER_FRAME / config.ROBOT_AUDIO_SAMPLE_RATE,
                    )
                except asyncio.TimeoutError:
                    pcm = self._silence_chunk()
                    playback_done = None
                if playback_done is not None and not playback_done.done():
                    playback_done.set_result(None)
                return self._build_frame(pcm)

            async def push_audio(
                self,
                wav_bytes: bytes,
                prefix_silence_sec: float = 0.0,
            ) -> tuple[float, asyncio.Future]:
                audio = AudioSegment.from_wav(io.BytesIO(wav_bytes))
                audio = audio.set_channels(config.ROBOT_AUDIO_CHANNELS).set_frame_rate(config.ROBOT_AUDIO_SAMPLE_RATE)
                if prefix_silence_sec > 0:
                    silence = AudioSegment.silent(
                        duration=int(prefix_silence_sec * 1000),
                        frame_rate=config.ROBOT_AUDIO_SAMPLE_RATE,
                    ).set_channels(config.ROBOT_AUDIO_CHANNELS)
                    audio = silence + audio
                pcm = np.array(audio.get_array_of_samples(), dtype=np.int16).reshape(-1, config.ROBOT_AUDIO_CHANNELS)
                playback_done = asyncio.get_running_loop().create_future()
                for i in range(0, len(pcm), config.ROBOT_SAMPLES_PER_FRAME):
                    chunk = pcm[i : i + config.ROBOT_SAMPLES_PER_FRAME]
                    if len(chunk) < config.ROBOT_SAMPLES_PER_FRAME:
                        pad = np.zeros(
                            (config.ROBOT_SAMPLES_PER_FRAME - len(chunk), config.ROBOT_AUDIO_CHANNELS),
                            dtype=np.int16,
                        )
                        chunk = np.vstack([chunk, pad])
                    is_last_chunk = i + config.ROBOT_SAMPLES_PER_FRAME >= len(pcm)
                    await self._queue.put((chunk, playback_done if is_last_chunk else None))
                return len(audio) / 1000.0, playback_done

        return QueuedAudioTrack()


    async def speak(self, text: str) -> bool:
        if self.state.text_mode:
            print("[TTS] Skipped in --text mode")
            return False
        if self.state.no_robot:
            print("[TTS] Skipped in --no-robot mode")
            return False
        async with self.state.speak_lock:
            self.state.tts_active.set()
            if not await self.connection.ensure_robot_connection():
                print("[TTS] Skipped because robot connection is down")
                self.state.tts_active.clear()
                return False
            spoken_text = sanitize_for_tts(text)
            if not spoken_text:
                print("[TTS] Skipped empty spoken text")
                self.state.tts_active.clear()
                return False
            try:
                if config.GENERATED_TTS_SETTLE_SEC > 0:
                    print(f"[TTS] Settling for {config.GENERATED_TTS_SETTLE_SEC:.1f}s...")
                    await asyncio.sleep(config.GENERATED_TTS_SETTLE_SEC)
                print("[TTS] Generating...")
                response = await asyncio.to_thread(
                    self.client.audio.speech.complete,
                    model="voxtral-mini-tts-2603",
                    input=spoken_text,
                    voice_id=config.TTS_VOICE_ID,
                    response_format="wav",
                )
                wav_bytes = base64.b64decode(response.audio_data)
                if not await self.connection.ensure_robot_connection():
                    print("[TTS] Skipped because robot connection dropped during TTS generation")
                    return False
                print("[TTS] Ready, uploading to AudioHub...")
                return await self.play_audiohub_uploaded_wav(
                    wav_bytes,
                    spoken_text,
                    prefix_silence_sec=config.PREFIX_SILENCE_SEC,
                )
            finally:
                self.state.tts_active.clear()


    async def play_wav_file(self, wav_path: Path, label: Optional[str] = None) -> bool:
        if self.state.text_mode:
            print("[AudioFile] Skipped in --text mode")
            return False
        if self.state.no_robot:
            print("[AudioFile] Skipped in --no-robot mode")
            return False
        if not wav_path.exists():
            raise FileNotFoundError(f"WAV file not found: {wav_path}")

        async with self.state.speak_lock:
            self.state.tts_active.set()
            if not await self.connection.ensure_robot_connection():
                print("[AudioFile] Skipped because robot connection is down")
                self.state.tts_active.clear()
                return False
            try:
                wav_bytes = wav_path.read_bytes()
                if label:
                    print(f"[AudioFile] {label}")
                print(f"[AudioFile] Uploading and playing {wav_path.name} via AudioHub...", flush=True)
                return await self.play_audiohub_uploaded_wav(wav_bytes, label or wav_path.stem)
            finally:
                self.state.tts_active.clear()


    async def play_knock_prompt_parallel(self, do_skill) -> dict:
        print("[Say] Knock knock, can I come in?")

        skill_task = asyncio.create_task(do_skill("Hello"))
        audiohub_task = asyncio.create_task(
            self.play_prompt_audio(
                label="Knock knock, can I come in?",
                fallback_uuid=config.KNOCK_KNOCK_AUDIO_UUID,
                fallback_duration_sec=config.KNOCK_KNOCK_AUDIO_DURATION_SEC,
            )
        )
        skill_outcome, audiohub_spoken = await asyncio.gather(
            skill_task,
            audiohub_task,
            return_exceptions=True,
        )

        if isinstance(skill_outcome, Exception):
            skill_result = {"status": "error", "message": str(skill_outcome)}
        else:
            try:
                skill_result = json.loads(skill_outcome)
            except Exception:
                skill_result = {"status": "error", "message": f"Invalid skill response: {skill_outcome}"}

        if isinstance(audiohub_spoken, Exception):
            print(f"[AudioHub] Parallel playback failed: {audiohub_spoken}", flush=True)
            audiohub_spoken = False

        knock_source = "audiohub"
        knock_spoken = bool(audiohub_spoken)

        return {
            "status": "ok" if knock_spoken and skill_result.get("status") == "ok" else "error",
            "spoken": knock_spoken,
            "message": "Knock knock, can I come in?",
            "skill": "Hello",
            "skill_result": skill_result,
            "source": knock_source,
            "uuid": config.KNOCK_KNOCK_AUDIO_UUID,
        }


    async def play_thank_you_prompt(self) -> dict:
        print("[Say] Thank you.")
        spoken = await self.play_prompt_audio(
            label="Thank you.",
            fallback_uuid=config.THANK_YOU_AUDIO_UUID,
            fallback_duration_sec=config.THANK_YOU_AUDIO_DURATION_SEC,
        )
        return {
            "status": "ok" if spoken else "error",
            "spoken": spoken,
            "message": "Thank you!",
            "source": "audiohub",
            "uuid": config.THANK_YOU_AUDIO_UUID,
        }

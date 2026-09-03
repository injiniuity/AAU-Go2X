import asyncio
import io
import threading
import time
import wave
from collections import deque
from typing import Optional

import numpy as np
from mistralai.client import models as mistral_models
from openwakeword.model import Model

from .. import config
from ..utils.logging_utils import print


class InputService:
    """Owns console input, wake-word capture, and speech transcription."""

    def __init__(self, *, state, client) -> None:
        self.state = state
        self.client = client

    def load_wakeword_model(self) -> Model:
        print(f"Loading wake word model: {config.WAKEWORD_MODEL_PATH}")
        return Model(wakeword_models=[config.WAKEWORD_MODEL_PATH], inference_framework="onnx")


    def keyboard_quit_watcher(self):
        import msvcrt

        while not self.state.stop_event.is_set():
            key = msvcrt.getwch()
            if key.lower() == "q":
                self.state.stop_event.set()
                print("\nQuit requested")
                break


    def chunk_rms(self, audio_float32: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(audio_float32)))) if len(audio_float32) else 0.0


    def numpy_to_wav_bytes(self, audio_np: np.ndarray) -> bytes:
        pcm = (audio_np * 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(config.SAMPLE_RATE)
            wf.writeframes(pcm.tobytes())
        return buf.getvalue()


    def transcribe(self, audio_np: np.ndarray) -> str:
        wav_bytes = self.numpy_to_wav_bytes(audio_np)
        result = self.client.audio.transcriptions.complete(
            model="voxtral-mini-latest",
            file=mistral_models.File(
                file_name="audio.wav",
                content=wav_bytes,
                content_type="audio/wav",
            ),
            language="en",
        )
        return result.text.strip()


    async def voice_loop(self, run_user_text):
        import sounddevice as sd

        wakeword_model = self.load_wakeword_model()
        self.state.command_queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        chunks_until_stop = max(1, int(config.SILENCE_SECONDS_TO_STOP * config.SAMPLE_RATE / config.WAKEWORD_CHUNK_SAMPLES))
        max_recording_chunks = max(1, int(config.MAX_COMMAND_SECONDS * config.SAMPLE_RATE / config.WAKEWORD_CHUNK_SAMPLES))

        threading.Thread(target=self.keyboard_quit_watcher, daemon=True).start()

        print("=== Go2 Voice Control + VLM (Wake Word) ===")
        print('Say "hey max", then speak your command. Press Q to quit.\n')

        def detector_worker():
            cooldown_until = 0.0
            detector_pause_reason: Optional[str] = None
            recording_active = False
            capture_mode = ""
            recorded_chunks: list[np.ndarray] = []
            speech_detected = False
            silence_chunks = 0
            pre_roll_chunks = max(1, int(config.PRE_ROLL_SECONDS * config.SAMPLE_RATE / config.WAKEWORD_CHUNK_SAMPLES))
            recent_audio = deque(maxlen=pre_roll_chunks)

            def reset_detector_state(clear_preroll: bool = False) -> None:
                nonlocal recording_active, capture_mode, recorded_chunks, speech_detected, silence_chunks
                recording_active = False
                capture_mode = ""
                recorded_chunks = []
                speech_detected = False
                silence_chunks = 0
                if clear_preroll:
                    recent_audio.clear()

            with sd.InputStream(
                samplerate=config.SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=config.WAKEWORD_CHUNK_SAMPLES,
            ) as stream:
                while not self.state.stop_event.is_set():
                    chunk, _ = stream.read(config.WAKEWORD_CHUNK_SAMPLES)
                    chunk_int16 = chunk[:, 0].copy()
                    chunk_float32 = chunk_int16.astype(np.float32) / 32768.0
                    recent_audio.append(chunk_float32.copy())
                    now = time.time()
                    passive_listen_active = self.state.passive_listen_queue is not None and now < self.state.passive_listen_deadline

                    pause_reason = None
                    if self.state.tts_active.is_set():
                        pause_reason = "tts"
                    elif not passive_listen_active and self.state.assistant_busy.is_set():
                        pause_reason = "assistant"
                    elif not passive_listen_active and now < self.state.wakeword_ignore_until:
                        pause_reason = "cooldown"

                    if pause_reason is not None:
                        if detector_pause_reason != pause_reason:
                            wakeword_model.reset()
                            reset_detector_state(clear_preroll=True)
                        detector_pause_reason = pause_reason
                        continue
                    if detector_pause_reason is not None:
                        wakeword_model.reset()
                        reset_detector_state(clear_preroll=True)
                        detector_pause_reason = None

                    if not recording_active:
                        if passive_listen_active:
                            rms = self.chunk_rms(chunk_float32)
                            if rms > config.SILENCE_THRESHOLD:
                                recording_active = True
                                capture_mode = "passive"
                                recorded_chunks = list(recent_audio)
                                speech_detected = True
                                silence_chunks = 0
                                print("[Door] Voice detected, recording reply...", flush=True)
                            continue

                        pred = wakeword_model.predict(chunk_int16)
                        score = pred[config.WAKEWORD_NAME]
                        if score > config.WAKEWORD_THRESHOLD and now >= cooldown_until:
                            cooldown_until = now + config.WAKEWORD_COOLDOWN_SECONDS
                            wakeword_model.reset()
                            recording_active = True
                            capture_mode = "wakeword"
                            recorded_chunks = list(recent_audio)
                            speech_detected = False
                            silence_chunks = 0
                            ts = time.strftime("%H:%M:%S")
                            print(f"[{ts}] Wake word detected (score={score:.3f})")
                            print("[Listening...] Speak your command", flush=True)
                        continue

                    recorded_chunks.append(chunk_float32)
                    rms = self.chunk_rms(chunk_float32)
                    if rms > config.SILENCE_THRESHOLD:
                        speech_detected = True
                        silence_chunks = 0
                    elif speech_detected:
                        silence_chunks += 1

                    should_stop = False
                    if speech_detected and silence_chunks >= chunks_until_stop:
                        should_stop = True
                    if len(recorded_chunks) >= max_recording_chunks:
                        should_stop = True
                    if not should_stop:
                        continue

                    recording_active = False
                    completed_mode = capture_mode
                    capture_mode = ""
                    audio_np = np.concatenate(recorded_chunks).astype(np.float32)
                    recorded_chunks = []

                    if not speech_detected or len(audio_np) == 0:
                        if completed_mode == "passive":
                            print("[Door] No reply detected")
                        else:
                            print("[Wake word] No command detected after wake word")
                        continue

                    if completed_mode == "passive":
                        target_queue = self.state.passive_listen_queue
                        if target_queue is not None:
                            loop.call_soon_threadsafe(target_queue.put_nowait, audio_np)
                    else:
                        loop.call_soon_threadsafe(self.state.command_queue.put_nowait, audio_np)

        threading.Thread(target=detector_worker, daemon=True).start()

        while not self.state.stop_event.is_set():
            try:
                audio_np = await asyncio.wait_for(self.state.command_queue.get(), timeout=0.2)
            except asyncio.TimeoutError:
                continue

            self.state.assistant_busy.set()
            try:
                print("[Transcribing...]", flush=True)
                text = await asyncio.to_thread(self.transcribe, audio_np)
            finally:
                self.state.assistant_busy.clear()

            if text:
                await run_user_text(text)
            else:
                print("[Nothing recognized]")


    async def text_loop(self, run_user_text):
        text_queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        print("=== Go2 LLM + VLM Control (Natural Language Text mode) ===")
        print("Type natural-language commands. They will be queued and processed in order.")
        print("Type 'q' to quit.\n")

        def text_input_worker():
            while not self.state.stop_event.is_set():
                try:
                    user_input = input("You: ")
                except EOFError:
                    self.state.stop_event.set()
                    loop.call_soon_threadsafe(text_queue.put_nowait, None)
                    break

                user_input = user_input.strip()
                if not user_input:
                    continue
                if user_input.lower() == "q":
                    self.state.stop_event.set()
                    loop.call_soon_threadsafe(text_queue.put_nowait, None)
                    break

                prompt_queue = self.state.text_mode_prompt_queue
                if prompt_queue is not None:
                    loop.call_soon_threadsafe(prompt_queue.put_nowait, user_input)
                    print(f"[{self.state.text_mode_prompt_label or 'Prompt'}] {user_input}", flush=True)
                    continue

                loop.call_soon_threadsafe(text_queue.put_nowait, user_input)
                print(f"[Queued] {user_input}", flush=True)

        threading.Thread(target=text_input_worker, daemon=True).start()

        while not self.state.stop_event.is_set():
            try:
                user_input = await asyncio.wait_for(text_queue.get(), timeout=0.2)
            except asyncio.TimeoutError:
                continue

            if user_input is None:
                break

            await run_user_text(user_input)
            print()

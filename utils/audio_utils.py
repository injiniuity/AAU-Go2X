"""Pure helpers for AudioHub records and WAV payloads."""

import io
import json
import re
import wave
from typing import Any, Optional

from pydub import AudioSegment

ROBOT_AUDIO_SAMPLE_RATE = 44100
ROBOT_AUDIO_CHANNELS = 2


def normalize_audio_name(text: str) -> str:
    return " ".join(str(text).strip().casefold().split())
def get_wav_duration_sec(wav_bytes: bytes) -> float:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        frame_rate = wf.getframerate()
        frame_count = wf.getnframes()
    if frame_rate <= 0:
        return 0.0
    return frame_count / frame_rate



def sanitize_audiohub_record_name(text: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", str(text).strip())
    safe = safe.strip("_")
    return safe[:48] or "tts"



def parse_audiohub_payload(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return value
        try:
            return parse_audiohub_payload(json.loads(text))
        except Exception:
            return value
    if isinstance(value, dict):
        parsed = {str(key): parse_audiohub_payload(child) for key, child in value.items()}
        nested = parsed.get("data")
        if isinstance(nested, dict) and "audio_list" in nested:
            return nested
        return parsed
    if isinstance(value, list):
        return [parse_audiohub_payload(child) for child in value]
    return value



def get_audiohub_record_uuid(record: dict[str, Any]) -> str:
    return str(record.get("unique_id", record.get("UNIQUE_ID", ""))).strip()



def get_audiohub_record_name(record: dict[str, Any]) -> str:
    return str(
        record.get(
            "file_name",
            record.get(
                "name",
                record.get("CUSTOM_NAME", record.get("custom_name", "")),
            ),
        )
    ).strip()



def get_audiohub_record_create_time(record: dict[str, Any]) -> int:
    value = record.get("create_time", record.get("CREATE_TIME", 0))
    try:
        return int(value)
    except Exception:
        return 0



def collect_audiohub_records(value: Any) -> list[dict[str, Any]]:
    parsed_value = parse_audiohub_payload(value)
    records: list[dict[str, Any]] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            keys = {str(key) for key in node.keys()}
            if any(
                key in keys
                for key in {
                    "unique_id",
                    "file_name",
                    "name",
                    "UNIQUE_ID",
                    "CUSTOM_NAME",
                    "custom_name",
                }
            ):
                records.append(node)
            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(parsed_value)

    unique_records: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for record in records:
        unique_id = get_audiohub_record_uuid(record)
        name = get_audiohub_record_name(record)
        key = f"{unique_id}|{name}|{json.dumps(record, sort_keys=True, default=str)}"
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique_records.append(record)
    return unique_records



def select_audiohub_record_by_name(records: list[dict[str, Any]], file_stem: str) -> Optional[dict[str, Any]]:
    target = normalize_audio_name(file_stem)
    candidates = []
    for record in records:
        record_name = get_audiohub_record_name(record)
        if normalize_audio_name(record_name) == target:
            candidates.append(record)
    if not candidates:
        return None

    def sort_key(record: dict[str, Any]) -> tuple[int, str]:
        return get_audiohub_record_create_time(record), get_audiohub_record_uuid(record)

    return sorted(candidates, key=sort_key)[-1]



def prepend_silence_to_wav_bytes(wav_bytes: bytes, silence_sec: float) -> bytes:
    if silence_sec <= 0:
        return wav_bytes
    audio = AudioSegment.from_wav(io.BytesIO(wav_bytes))
    silence = AudioSegment.silent(
        duration=int(silence_sec * 1000),
        frame_rate=audio.frame_rate or ROBOT_AUDIO_SAMPLE_RATE,
    ).set_channels(audio.channels or ROBOT_AUDIO_CHANNELS)
    combined = silence + audio
    output = io.BytesIO()
    combined.export(output, format="wav")
    return output.getvalue()



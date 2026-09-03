import asyncio
import base64
import io
import json
import logging
import sys
import threading
import time
import wave
from collections import deque
from pathlib import Path
from typing import Any, Optional

import numpy as np
from mistralai.client import Mistral
from mistralai.client import models as mistral_models
from openwakeword.model import Model
from pydub import AudioSegment

from . import config
from .utils.audio_utils import (
    collect_audiohub_records,
    get_audiohub_record_create_time,
    get_audiohub_record_name,
    get_audiohub_record_uuid,
    get_wav_duration_sec,
    parse_audiohub_payload,
    prepend_silence_to_wav_bytes,
    sanitize_audiohub_record_name,
    select_audiohub_record_by_name,
)
from .llm.assistant_loop import run_user_request
from .llm.assistant_actions import AssistantActions
from .services.audio_service import AudioService
from .services.connection_service import RobotConnectionService
from .services.door_entry_service import DoorEntryService
from .services.input_service import InputService
from .utils.logging_utils import print
from .services.map_service import MapService
from .services.name_matching_service import NameMatchingService
from .services.navigation_service import NavigationService
from .services.person_presence_service import PersonPresenceService
from .utils.navigation_planner import door_approach_sequence, normalize_yaw, with_rotated_yaw
from .llm.prompts import SYSTEM_PROMPT
from .state import RuntimeState
from .utils.text_utils import (
    ascii_safe_text,
    normalize_name_key,
    quat_to_yaw,
    sanitize_for_json,
    sanitize_for_tts,
)
from .llm.tool_schemas import build_tool_schemas
from .utils.vision_utils import (
    encode_frame_as_jpeg_data_url,
)
from .services.vision_service import VisionService

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    from g2p_en import G2p
except ImportError:
    G2p = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from unitree_webrtc_connect.constants import RTC_TOPIC, SPORT_CMD
from unitree_webrtc_connect.webrtc_driver import (
    UnitreeWebRTCConnection,
    WebRTCConnectionMethod,
)
from unitree_webrtc_connect.webrtc_audiohub import WebRTCAudioHub

logging.getLogger("aiortc.codecs.h264").setLevel(logging.ERROR)
client = Mistral(api_key=config.MISTRAL_API_KEY)

state = RuntimeState()
name_matcher = NameMatchingService(
    state=state,
    model_id=config.NAME_MATCHER_MODEL_ID,
    g2p_cls=G2p,
    sentence_transformer_cls=SentenceTransformer,
)
map_service = MapService(state=state, name_matcher=name_matcher)
vision = VisionService(
    yolo_cls=YOLO,
    yolo_model_name=config.FIND_PERSON_YOLO_MODEL,
    yolo_target_classes=config.FIND_PERSON_YOLO_CLASSES,
)
connection = RobotConnectionService(
    state=state,
    connection_cls=UnitreeWebRTCConnection,
    connection_method=WebRTCConnectionMethod,
    robot_ip=config.ROBOT_IP,
    initial_retries=config.INITIAL_CONNECT_RETRIES,
    retry_delay_sec=config.INITIAL_CONNECT_RETRY_DELAY_SEC,
    build_audio_track=lambda: audio.build_audio_track(),
    register_streams_and_callbacks=lambda active_conn: register_robot_callbacks(active_conn),
    load_active_map_id=lambda: map_service.load_active_map_id(),
    load_named_locations=lambda: map_service.load_named_locations(),
    start_localization_once=lambda: navigation.start_localization_once(),
)
navigation = NavigationService(
    state=state,
    connection=connection,
    map_service=map_service,
)
audio = AudioService(state=state, connection=connection, client=client)
input_service = InputService(state=state, client=client)

ROOM_BY_LOCATION_KEY = map_service.room_by_location_key

async def complete_chat_async(**kwargs):
    return await asyncio.to_thread(client.chat.complete, **kwargs)

person_presence = PersonPresenceService(
    state=state,
    connection=connection,
    navigation=navigation,
    vision=vision,
    complete_chat=complete_chat_async,
)
door_entry = DoorEntryService(
    state=state,
    map_service=map_service,
    navigation=navigation,
    audio=audio,
    input_service=input_service,
    complete_chat=complete_chat_async,
)
actions = AssistantActions(
    state=state,
    connection=connection,
    navigation=navigation,
    map_service=map_service,
    vision=vision,
    audio=audio,
    door_entry=door_entry,
    person_presence=person_presence,
    complete_chat=complete_chat_async,
)

def register_robot_callbacks(active_conn) -> None:
    active_conn.video.add_track_callback(vision.recv_camera_stream)
    active_conn.video.switchVideoChannel(True)
    active_conn.datachannel.pub_sub.subscribe(
        RTC_TOPIC["LIDAR_MAPPING_SERVER_LOG"], navigation.slam_server_log_callback
    )
    active_conn.datachannel.pub_sub.subscribe(
        RTC_TOPIC["LIDAR_LOCALIZATION_ODOM"], navigation.pose_callback
    )
    active_conn.datachannel.pub_sub.subscribe(
        RTC_TOPIC["LIDAR_MAPPING_ODOM"], navigation.pose_callback
    )
    active_conn.datachannel.pub_sub.subscribe(
        RTC_TOPIC["AUDIO_HUB_PLAY_STATE"], audio.audiohub_play_state_callback
    )

tools = build_tool_schemas(config.SKILLS)

async def run(user_text: str) -> str:
    state.assistant_busy.set()
    print(f"\n[User]: {user_text}")
    try:
        return await run_user_request(
            user_text,
            complete_chat=complete_chat_async,
            model=config.LLM_MODEL,
            system_prompt=SYSTEM_PROMPT,
            tools=tools,
            handlers={
                "say_message": actions.say_message,
                "deliver_message_to_person": actions.deliver_message_to_person,
                "do_skill": actions.do_skill,
                "describe_view": actions.describe_view,
                "go_to": actions.go_to,
                "find_person": lambda description: person_presence.find_person(description, actions.do_skill),
                "check_seat_and_report_back": actions.check_seat_and_report_back,
            },
            speak=audio.speak,
        )
    finally:
        state.wakeword_ignore_until = time.time() + config.POST_TTS_IGNORE_SECONDS
        state.assistant_busy.clear()

async def main():

    text_mode = "--text" in sys.argv
    state.text_mode = text_mode
    state.no_robot = "--no-robot" in sys.argv

    map_service.load_named_locations()
    await asyncio.to_thread(name_matcher.preload)

    if state.no_robot:
        print("(--no-robot: TTS through PC speaker)\n")
    else:
        print("Connecting to robot...")
        state.conn = await connection.connect_with_retries()
        if not text_mode:
            state.conn.audio.switchAudioChannel(True)
            await connection.ensure_outgoing_audio_track(force_rebuild=True)
        register_robot_callbacks(state.conn)
        map_service.load_active_map_id()
        await navigation.start_localization_once()
        if config.CAMERA_PREVIEW_ENABLED:
            state.camera_preview_task = asyncio.create_task(vision.camera_preview_loop())
        print("Robot connected!\n")

    try:
        if text_mode:
            await input_service.text_loop(run)
        else:
            await input_service.voice_loop(run)
    finally:
        state.stop_event.set()
        if state.camera_preview_task is not None:
            state.camera_preview_task.cancel()
            try:
                await state.camera_preview_task
            except asyncio.CancelledError:
                pass
        if state.conn is not None:
            try:
                await navigation.stop_active_slam_modules_on_shutdown()
            except Exception as e:
                print(f"[Shutdown] SLAM cleanup error: {e}", flush=True)
            try:
                await state.conn.disconnect()
            except Exception as e:
                print(f"[WebRTC] Disconnect error: {e}", flush=True)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nQuit")
        sys.exit(0)

import os
import sys
from pathlib import Path

from dotenv import load_dotenv


CURRENT_WORKDIR = str(Path.cwd().resolve())
SCRIPT_PATH = Path(__file__).resolve()
PACKAGE_DIR = SCRIPT_PATH.parent
SCRIPT_DIR = str(PACKAGE_DIR.parent)
PROJECT_ROOT_DIR = str(PACKAGE_DIR.parent.parent)
ENV_FILE = Path(PROJECT_ROOT_DIR) / ".env"
if not ENV_FILE.exists():
    ENV_FILE = Path(PROJECT_ROOT_DIR).parent / ".env"

LOCAL_WEBRTC_SRC = Path(PROJECT_ROOT_DIR) / "unitree_webrtc_connect"
if str(LOCAL_WEBRTC_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_WEBRTC_SRC))
sys.path = [
    entry
    for entry in sys.path
    if entry not in {"", CURRENT_WORKDIR}
    or Path(entry).resolve()
    in {Path(SCRIPT_DIR), Path(PROJECT_ROOT_DIR), LOCAL_WEBRTC_SRC}
]

load_dotenv(ENV_FILE)

MISTRAL_API_KEY = os.environ["MISTRAL_API_KEY"]
# --no-robot mode never opens a robot connection, so an IP must not prevent it
# from starting. The connection service validates the value when it is needed.
ROBOT_IP = os.environ.get("UNITREE_ROBOT_IP", "").strip()

SAMPLE_RATE = 16000
ROBOT_AUDIO_SAMPLE_RATE = 44100
ROBOT_AUDIO_CHANNELS = 2
ROBOT_SAMPLES_PER_FRAME = 882
PREFIX_SILENCE_SEC = 0
AUDIO_PLAYBACK_TAIL_SEC = 0.25
GENERATED_TTS_SETTLE_SEC = 0.0
TTS_VOICE_ID = "c69964a6-ab8b-4f8a-9465-ec0925096ec8"
AUDIOHUB_UPLOAD_DIR = Path(SCRIPT_DIR) / "tts_runtime_uploads"
AUDIOHUB_UPLOAD_POLL_ATTEMPTS = 6
AUDIOHUB_UPLOAD_POLL_DELAY_SEC = 0.5

LLM_MODEL = "mistral-large-latest"
VLM_MODEL = "pixtral-12b-latest"

FIND_PERSON_YOLO_MODEL = "yolov8n.pt"
FIND_PERSON_YOLO_CLASSES = ("person", "chair")

VERBOSE_LOGS = "--log" in sys.argv
CAMERA_PREVIEW_ENABLED = True

DEFAULT_LOG_PREFIXES = (
    "[User]",
    "[Tool]",
    "[Result]",
    "[Robot]",
    "[Door]",
    "[Say]",
    "[NameMatch]",
    "[WebRTC]",
    "[Shutdown]",
    "[Wake word]",
    "[Listening...]",
    "[Transcribing...]",
    "[Nothing recognized]",
    "[Queued]",
    "[Door response]",
)
DEFAULT_LOG_STARTS = (
    "=== Go2 Voice Control + VLM (Wake Word) ===",
    'Say "hey max", then speak your command. Press Q to quit.',
    "=== Go2 LLM + VLM Control (Natural Language Text mode) ===",
    "Type natural-language commands. They will be queued and processed in order.",
    "Type 'q' to quit.",
    "Connecting to robot...",
    "Robot connected!",
    "Door response: ",
    "Loading wake word model:",
    "Quit requested",
    "Quit",
    "(--no-robot:",
)

WAKEWORD_MODEL_PATH = str(Path(SCRIPT_DIR) / "hey_max.onnx")
WAKEWORD_NAME = "hey_max"
WAKEWORD_THRESHOLD = 0.5
WAKEWORD_CHUNK_SAMPLES = 1280
SILENCE_THRESHOLD = 0.02
SILENCE_SECONDS_TO_STOP = 1.2
MAX_COMMAND_SECONDS = 8.0
WAKEWORD_COOLDOWN_SECONDS = 1.5
POST_TTS_IGNORE_SECONDS = 4.0
PRE_ROLL_SECONDS = 0.4

ENTRY_PERMISSION_TIMEOUT_SEC = 20.0
ROOM_ENTRY_REQUIRE_PERMISSION = True

INITIAL_CONNECT_RETRIES = 3
INITIAL_CONNECT_RETRY_DELAY_SEC = 8.0

LOCALIZATION_INIT_MAX_ATTEMPTS = 3
LOCALIZATION_INIT_STATE_TIMEOUT_SEC = 5.0
LOCALIZATION_POSE_SETTLE_SEC = 3.0

KNOCK_KNOCK_AUDIO_UUID = "4a9c9606-3e5e-48de-9320-3198f2cf6751"
THANK_YOU_AUDIO_UUID = "be164bf7-985f-476b-9e9f-65980ac987a2"
KNOCK_KNOCK_AUDIO_DURATION_SEC = 4.5
THANK_YOU_AUDIO_DURATION_SEC = 3.6

NAVIGATION_MAX_ATTEMPTS = 5
NAVIGATION_RETRY_DELAY_SEC = 1.0
NAVIGATION_FIRST_STATE_TIMEOUT_SEC = 5.0
NAVIGATION_ALREADY_AT_TARGET_DISTANCE_M = 0.18
NAV_FAILURE_STATES = {
    "NO_PATH",
    "TIMEOUT",
    "GOAL_OCCUPIED",
    "FAILURE",
    "TIMEOUT_POINTCLOUD",
    "ABNORMAL",
}

DOOR_REVERSE_MAX_DISTANCE_M = 0.45
DOOR_REVERSE_DURATION_SEC = 1.0
DOOR_REVERSE_PREPARE_SEC = 0.8
DOOR_REVERSE_WIRELESS_LY = 1.0
DOOR_REVERSE_COMMAND_INTERVAL_SEC = 0.02
DOOR_REVERSE_SETTLE_SEC = 0.4
DOOR_REVERSE_TRIGGER_DISTANCE_M = 0.15

ROOM_ENTRY_CONFIG = {
    "210": {
        "doors": ["door210"],
        "members": ["Initial", "Dimitris", "Chen", "Jini"],
    },
    "209": {
        "doors": ["door209-1", "door209-2"],
        "members": ["Tristan", "Samuel", "Martin", "BYERN", "Byern", "point2"],
        "member_doors": {
            "Tristan": "door209-1",
            "Samuel": "door209-1",
            "Martin": "door209-2",
            "BYERN": "door209-2",
            "Byern": "door209-2",
            "point2": "door209-2",
        },
    },
    "208": {
        "doors": ["door208"],
        "members": [
            "Bret",
            "Filip",
            "Christian",
            "Moshin",
            "Mohsin",
            "Mads",
            "David",
            "Sebastian",
            "Rebeca",
        ],
    },
}
ROOM_TRANSITION_DOOR_OVERRIDES = {
    ("209", "208"): ["door209-2"],
}

SKILLS = [
    "Sit",
    "StandUp",
    "StandDown",
    "Hello",
    "Stretch",
    "Dance1",
    "WiggleHips",
    "FingerHeart",
    "Scrape",
]

OFFICE210_FILE_CANDIDATES = [
    Path(SCRIPT_DIR) / "entire_office.json",
    Path(SCRIPT_DIR) / "office210.json",
]
DEFAULT_OFFICE210_NAMES = ["Initial", "Dimitris", "Chen", "Jini"]

LOCATION_ARRIVAL_THRESHOLD = 0.45
LOCATION_TIMEOUT_SEC = 90.0
POINT_ONE_CLOSE_ENOUGH_THRESHOLD_M = 0.30
CORRIDOR_APPROACH_CHAIN = ["b", "d"]
SLAM_DEBUG_LOGS = True
RETURN_TO_NAMED_LOCATION_MAX_DISTANCE = 0.8
RETURN_TO_POSE_MAX_AGE_SEC = 5.0

NAME_MATCHER_MODEL_ID = "BAAI/bge-m3"

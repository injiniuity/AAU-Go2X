import asyncio
import json
import sys
from pathlib import Path

def find_local_webrtc_source() -> Path:
    """Find an adjacent clone of the external WebRTC dependency."""
    for directory in Path(__file__).resolve().parents:
        candidate = directory / "unitree_webrtc_connect"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not find unitree_webrtc_connect. Clone it next to this project or one of its parent folders."
    )


LOCAL_WEBRTC_SRC = find_local_webrtc_source()
if str(LOCAL_WEBRTC_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_WEBRTC_SRC))

from unitree_webrtc_connect import UnitreeWebRTCConnection, WebRTCConnectionMethod
from unitree_webrtc_connect.webrtc_audiohub import WebRTCAudioHub


async def get_list(hub):
    resp = await hub.get_audio_list()
    data_str = resp.get("data", {}).get("data", "{}")
    return json.loads(data_str).get("audio_list", [])


async def main():
    conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalAP)
    await conn.connect()
    hub = WebRTCAudioHub(conn, None)

    while True:
        audio_list = await get_list(hub)
        print(f"\n=== Robot audio list ({len(audio_list)} files) ===")
        for i, a in enumerate(audio_list):
            print(f"  {i+1}. {a['CUSTOM_NAME']}  ({a['UNIQUE_ID']})")

        print("\n  d <number> : delete")
        print("  u <file path> : upload  (e.g. u Jini/tts/StandUp.wav)")
        print("  q : quit\n")

        cmd = await asyncio.to_thread(input, "> ")
        cmd = cmd.strip()

        if cmd.lower() == "q":
            break
        elif cmd.lower().startswith("d "):
            try:
                idx = int(cmd.split()[1]) - 1
                target = audio_list[idx]
                await hub.delete_record(target["UNIQUE_ID"])
                print(f"  Deleted: {target['CUSTOM_NAME']}")
            except Exception as e:
                print(f"  Error: {e}")
        elif cmd.lower().startswith("u "):
            path = cmd[2:].strip()
            print(f"  Uploading: {path}")
            try:
                await hub.upload_audio_file(path)
                print("  Upload complete")
            except Exception as e:
                print(f"  Error: {e}")
        else:
            print("  Unknown command")

asyncio.run(main())

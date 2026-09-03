import base64
import os
from pathlib import Path

from dotenv import load_dotenv
from mistralai.client import Mistral


OUTPUT_DIR = Path(__file__).resolve().parent.parent / "tts"
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
if not ENV_FILE.exists():
    ENV_FILE = ENV_FILE.parent.parent / ".env"
TTS_MODEL = "voxtral-mini-tts-2603"
VOICE_ID = "c69964a6-ab8b-4f8a-9465-ec0925096ec8"  # same voice as LLM_fuctioncalling.py

TEXT = "Knock Knock! can I come in?"


def main() -> None:
    load_dotenv(ENV_FILE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise EnvironmentError("MISTRAL_API_KEY is not set.")
    client = Mistral(api_key=api_key)

    print(f"[tts] {TEXT}")
    response = client.audio.speech.complete(
        model=TTS_MODEL,
        input=TEXT,
        voice_id=VOICE_ID,
        response_format="wav",
    )
    out_path = OUTPUT_DIR / "knock_knock.wav"
    out_path.write_bytes(base64.b64decode(response.audio_data))
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()

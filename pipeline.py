"""
podcast-dubber pipeline
=======================
Complete processing pipeline:
  1. yt-dlp  -> extract audio from YouTube
  2. Whisper -> transcribe with timestamps
  3. GPT-4o  -> translate segments to Simplified Chinese
  4. TTS     -> generate Chinese speech per segment
  5. pydub   -> align & stitch into final MP3
"""

import json
import math
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import openai
from pydub import AudioSegment

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

MAX_SPEED = 1.2  # maximum atempo factor before we truncate


def _client() -> openai.OpenAI:
    return openai.OpenAI(api_key=OPENAI_API_KEY)


def _update(progress_cb: Callable | None, stage: str, pct: int):
    if progress_cb:
        progress_cb(stage, pct)


# ---------------------------------------------------------------------------
# Step 1 - Download audio with yt-dlp
# ---------------------------------------------------------------------------

def download_audio(youtube_url: str, tmp_dir: str) -> str:
    """Download YouTube audio as 128 kbps MP3. Returns path to mp3 file."""
    # Upgrade yt-dlp to latest version to avoid download failures
    subprocess.run(
        ["pip", "install", "--upgrade", "yt-dlp"],
        capture_output=True, text=True
    )

    output_template = os.path.join(tmp_dir, "source.%(ext)s")
    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format", "mp3",
        "--audio-quality", "128K",
        "--no-playlist",
        "--js-runtimes", "nodejs",
        "-o", output_template,
        youtube_url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"yt-dlp failed (exit {result.returncode}): {result.stderr[:500]}"
        )

    mp3_path = os.path.join(tmp_dir, "source.mp3")
    if not os.path.exists(mp3_path):
        for f in os.listdir(tmp_dir):
            if f.endswith(".mp3"):
                mp3_path = os.path.join(tmp_dir, f)
                break
    return mp3_path


# ---------------------------------------------------------------------------
# Step 2 - Transcribe with Whisper
# ---------------------------------------------------------------------------

def transcribe_audio(mp3_path: str) -> list[dict]:
    """
    Transcribe audio using OpenAI Whisper API (whisper-1, verbose_json).
    Returns list of segments: [{id, start, end, text}, ...]
    """
    client = _client()
    with open(mp3_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )

    segments = []
    for seg in response.segments:
        segments.append({
            "id": seg.id if hasattr(seg, "id") else seg.get("id", len(segments)),
            "start": seg.start if hasattr(seg, "start") else seg["start"],
            "end": seg.end if hasattr(seg, "end") else seg["end"],
            "text": seg.text if hasattr(seg, "text") else seg["text"],
        })
    return segments


# ---------------------------------------------------------------------------
# Step 3 - Translate with GPT-4o (batched)
# ---------------------------------------------------------------------------

TRANSLATE_SYSTEM = (
    "You are a professional translator. Translate the following segments "
    "into Simplified Chinese. Return ONLY a JSON object with a key "
    '"segments" containing an array where each element '
    'has "id" (integer) and "text" (translated string). '
    "Preserve the original segment ids. Do not add any commentary."
)

BATCH_SIZE = 30


def _extract_translations(result) -> list[dict]:
    """Robustly extract translation array from GPT response."""
    if isinstance(result, list):
        return result
    if isinstance(result, dict):
        # Try known keys first
        for key in ("segments", "translations", "data", "results"):
            val = result.get(key)
            if isinstance(val, list):
                return val
        # Fallback: find first list value in the dict
        for val in result.values():
            if isinstance(val, list):
                return val
    raise ValueError(f"Cannot extract translations from response: {type(result)}")


def translate_segments(segments: list[dict], progress_cb: Callable | None = None) -> list[dict]:
    """Translate segment texts to Simplified Chinese using GPT-4o in batches."""
    client = _client()
    translated: dict[int, str] = {}
    total_batches = math.ceil(len(segments) / BATCH_SIZE)

    for batch_idx in range(total_batches):
        start = batch_idx * BATCH_SIZE
        batch = segments[start : start + BATCH_SIZE]
        payload = [{"id": s["id"], "text": s["text"]} for s in batch]

        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model="gpt-4o",
                    temperature=0.3,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": TRANSLATE_SYSTEM},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                    ],
                )
                content = resp.choices[0].message.content
                result = json.loads(content)
                arr = _extract_translations(result)
                for item in arr:
                    translated[item["id"]] = item["text"]
                break
            except Exception as e:
                if attempt == 2:
                    raise RuntimeError(f"Translation failed after 3 attempts: {e}")
                time.sleep(2)

        pct = int((batch_idx + 1) / total_batches * 100)
        _update(progress_cb, "translating", pct)

    # Merge translations back
    result_segments = []
    for seg in segments:
        result_segments.append({
            **seg,
            "translated": translated.get(seg["id"], seg["text"]),
        })
    return result_segments


# ---------------------------------------------------------------------------
# Step 4 - Generate TTS per segment
# ---------------------------------------------------------------------------

def generate_tts(segments: list[dict], tmp_dir: str, progress_cb: Callable | None = None) -> list[dict]:
    """Generate Chinese TTS for each segment. Returns segments with 'tts_path' added."""
    client = _client()
    total = len(segments)

    for i, seg in enumerate(segments):
        text = seg["translated"]
        if not text.strip():
            seg["tts_path"] = None
            continue

        out_path = os.path.join(tmp_dir, f"tts_{seg['id']}.mp3")
        for attempt in range(3):
            try:
                response = client.audio.speech.create(
                    model="tts-1",
                    voice="nova",
                    input=text,
                )
                response.stream_to_file(out_path)
                break
            except Exception as e:
                if attempt == 2:
                    raise RuntimeError(f"TTS generation failed for segment {seg['id']}: {e}")
                time.sleep(2)

        seg["tts_path"] = out_path
        pct = int((i + 1) / total * 100)
        _update(progress_cb, "generating_tts", pct)

    return segments


# ---------------------------------------------------------------------------
# Step 5 - Align and stitch audio
# ---------------------------------------------------------------------------

def _speed_up(audio: AudioSegment, factor: float, tmp_dir: str) -> AudioSegment:
    """Speed up audio using ffmpeg atempo filter."""
    in_path = os.path.join(tmp_dir, "speed_in.mp3")
    out_path = os.path.join(tmp_dir, "speed_out.mp3")
    audio.export(in_path, format="mp3")

    cmd = [
        "ffmpeg", "-y", "-i", in_path,
        "-filter:a", f"atempo={factor:.4f}",
        "-vn", out_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return AudioSegment.from_mp3(out_path)


def stitch_audio(segments: list[dict], tmp_dir: str, progress_cb: Callable | None = None) -> AudioSegment:
    """
    Align TTS segments to original timestamps and stitch into a single audio.
    """
    if not segments:
        return AudioSegment.silent(duration=0)

    total_duration_ms = int(segments[-1]["end"] * 1000)
    final = AudioSegment.silent(duration=total_duration_ms)
    total = len(segments)

    for i, seg in enumerate(segments):
        tts_path = seg.get("tts_path")
        if not tts_path or not os.path.exists(tts_path):
            continue

        start_ms = int(seg["start"] * 1000)
        end_ms = int(seg["end"] * 1000)
        slot_ms = end_ms - start_ms

        if slot_ms <= 0:
            continue

        tts_audio = AudioSegment.from_mp3(tts_path)
        tts_len = len(tts_audio)

        if tts_len <= slot_ms:
            final = final.overlay(tts_audio, position=start_ms)
        else:
            speed_factor = tts_len / slot_ms
            if speed_factor <= MAX_SPEED:
                sped = _speed_up(tts_audio, speed_factor, tmp_dir)
                final = final.overlay(sped[:slot_ms], position=start_ms)
            else:
                sped = _speed_up(tts_audio, MAX_SPEED, tmp_dir)
                final = final.overlay(sped[:slot_ms], position=start_ms)

        pct = int((i + 1) / total * 100)
        _update(progress_cb, "stitching", pct)

    return final


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(youtube_url: str, job_id: str, progress_cb: Callable | None = None) -> str:
    """
    Execute the full dubbing pipeline.
    Returns path to the final MP3 file.
    """
    with tempfile.TemporaryDirectory(prefix="dubber_") as tmp_dir:
        # Step 1: Download
        _update(progress_cb, "downloading", 0)
        mp3_path = download_audio(youtube_url, tmp_dir)
        _update(progress_cb, "downloading", 100)

        # Step 2: Transcribe
        _update(progress_cb, "transcribing", 0)
        segments = transcribe_audio(mp3_path)
        _update(progress_cb, "transcribing", 100)

        # Step 3: Translate
        _update(progress_cb, "translating", 0)
        segments = translate_segments(segments, progress_cb)

        # Step 4: TTS
        _update(progress_cb, "generating_tts", 0)
        segments = generate_tts(segments, tmp_dir, progress_cb)

        # Step 5: Stitch
        _update(progress_cb, "stitching", 0)
        final_audio = stitch_audio(segments, tmp_dir, progress_cb)

        # Step 6: Export
        _update(progress_cb, "exporting", 0)
        output_path = str(OUTPUT_DIR / f"{job_id}.mp3")
        final_audio.export(output_path, format="mp3", bitrate="128k")
        _update(progress_cb, "exporting", 100)

    return output_path
"""
podcast-dubber pipeline
=======================
Complete processing pipeline:
  1. yt-dlp  -> extract audio from YouTube
  2. Whisper -> transcribe with timestamps
  3. GPT-4o  -> translate segments to Simplified Chinese
  4. TTS     -> generate Chinese speech per segment
  5. pydub   -> align & stitch into final MP3
"""

import json
import math
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import openai
from pydub import AudioSegment

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

MAX_SPEED = 1.2  # maximum atempo factor before we truncate


def _client() -> openai.OpenAI:
    return openai.OpenAI(api_key=OPENAI_API_KEY)


def _update(progress_cb: Callable | None, stage: str, pct: int):
    if progress_cb:
        progress_cb(stage, pct)


# ---------------------------------------------------------------------------
# Step 1 - Download audio with yt-dlp
# ---------------------------------------------------------------------------

def download_audio(youtube_url: str, tmp_dir: str) -> str:
    """Download YouTube audio as 128 kbps MP3. Returns path to mp3 file."""
    # Upgrade yt-dlp to latest version to avoid download failures
    subprocess.run(
        ["pip", "install", "--upgrade", "yt-dlp"],
        capture_output=True, text=True
    )

    output_template = os.path.join(tmp_dir, "source.%(ext)s")
    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format", "mp3",
        "--audio-quality", "128K",
        "--no-playlist",
        "-o", output_template,
        youtube_url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"yt-dlp failed (exit {result.returncode}): {result.stderr[:500]}"
        )

    mp3_path = os.path.join(tmp_dir, "source.mp3")
    if not os.path.exists(mp3_path):
        for f in os.listdir(tmp_dir):
            if f.endswith(".mp3"):
                mp3_path = os.path.join(tmp_dir, f)
                break
    return mp3_path


# ---------------------------------------------------------------------------
# Step 2 - Transcribe with Whisper
# ---------------------------------------------------------------------------

def transcribe_audio(mp3_path: str) -> list[dict]:
    """
    Transcribe audio using OpenAI Whisper API (whisper-1, verbose_json).
    Returns list of segments: [{id, start, end, text}, ...]
    """
    client = _client()
    with open(mp3_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )

    segments = []
    for seg in response.segments:
        segments.append({
            "id": seg.id if hasattr(seg, "id") else seg.get("id", len(segments)),
            "start": seg.start if hasattr(seg, "start") else seg["start"],
            "end": seg.end if hasattr(seg, "end") else seg["end"],
            "text": seg.text if hasattr(seg, "text") else seg["text"],
        })
    return segments


# ---------------------------------------------------------------------------
# Step 3 - Translate with GPT-4o (batched)
# ---------------------------------------------------------------------------

TRANSLATE_SYSTEM = (
    "You are a professional translator. Translate the following segments "
    "into Simplified Chinese. Return ONLY a JSON object with a key "
    '"segments" containing an array where each element '
    'has "id" (integer) and "text" (translated string). '
    "Preserve the original segment ids. Do not add any commentary."
)

BATCH_SIZE = 30


def _extract_translations(result) -> list[dict]:
    """Robustly extract translation array from GPT response."""
    if isinstance(result, list):
        return result
    if isinstance(result, dict):
        # Try known keys first
        for key in ("segments", "translations", "data", "results"):
            val = result.get(key)
            if isinstance(val, list):
                return val
        # Fallback: find first list value in the dict
        for val in result.values():
            if isinstance(val, list):
                return val
    raise ValueError(f"Cannot extract translations from response: {type(result)}")


def translate_segments(segments: list[dict], progress_cb: Callable | None = None) -> list[dict]:
    """Translate segment texts to Simplified Chinese using GPT-4o in batches."""
    client = _client()
    translated: dict[int, str] = {}
    total_batches = math.ceil(len(segments) / BATCH_SIZE)

    for batch_idx in range(total_batches):
        start = batch_idx * BATCH_SIZE
        batch = segments[start : start + BATCH_SIZE]
        payload = [{"id": s["id"], "text": s["text"]} for s in batch]

        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model="gpt-4o",
                    temperature=0.3,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": TRANSLATE_SYSTEM},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                    ],
                )
                content = resp.choices[0].message.content
                result = json.loads(content)
                arr = _extract_translations(result)
                for item in arr:
                    translated[item["id"]] = item["text"]
                break
            except Exception as e:
                if attempt == 2:
                    raise RuntimeError(f"Translation failed after 3 attempts: {e}")
                time.sleep(2)

        pct = int((batch_idx + 1) / total_batches * 100)
        _update(progress_cb, "translating", pct)

    # Merge translations back
    result_segments = []
    for seg in segments:
        result_segments.append({
            **seg,
            "translated": translated.get(seg["id"], seg["text"]),
        })
    return result_segments


# ---------------------------------------------------------------------------
# Step 4 - Generate TTS per segment
# ---------------------------------------------------------------------------

def generate_tts(segments: list[dict], tmp_dir: str, progress_cb: Callable | None = None) -> list[dict]:
    """Generate Chinese TTS for each segment. Returns segments with 'tts_path' added."""
    client = _client()
    total = len(segments)

    for i, seg in enumerate(segments):
        text = seg["translated"]
        if not text.strip():
            seg["tts_path"] = None
            continue

        out_path = os.path.join(tmp_dir, f"tts_{seg['id']}.mp3")
        for attempt in range(3):
            try:
                response = client.audio.speech.create(
                    model="tts-1",
                    voice="nova",
                    input=text,
                )
                response.stream_to_file(out_path)
                break
            except Exception as e:
                if attempt == 2:
                    raise RuntimeError(f"TTS generation failed for segment {seg['id']}: {e}")
                time.sleep(2)

        seg["tts_path"] = out_path
        pct = int((i + 1) / total * 100)
        _update(progress_cb, "generating_tts", pct)

    return segments


# ---------------------------------------------------------------------------
# Step 5 - Align and stitch audio
# ---------------------------------------------------------------------------

def _speed_up(audio: AudioSegment, factor: float, tmp_dir: str) -> AudioSegment:
    """Speed up audio using ffmpeg atempo filter."""
    in_path = os.path.join(tmp_dir, "speed_in.mp3")
    out_path = os.path.join(tmp_dir, "speed_out.mp3")
    audio.export(in_path, format="mp3")

    cmd = [
        "ffmpeg", "-y", "-i", in_path,
        "-filter:a", f"atempo={factor:.4f}",
        "-vn", out_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return AudioSegment.from_mp3(out_path)


def stitch_audio(segments: list[dict], tmp_dir: str, progress_cb: Callable | None = None) -> AudioSegment:
    """
    Align TTS segments to original timestamps and stitch into a single audio.
    """
    if not segments:
        return AudioSegment.silent(duration=0)

    total_duration_ms = int(segments[-1]["end"] * 1000)
    final = AudioSegment.silent(duration=total_duration_ms)
    total = len(segments)

    for i, seg in enumerate(segments):
        tts_path = seg.get("tts_path")
        if not tts_path or not os.path.exists(tts_path):
            continue

        start_ms = int(seg["start"] * 1000)
        end_ms = int(seg["end"] * 1000)
        slot_ms = end_ms - start_ms

        if slot_ms <= 0:
            continue

        tts_audio = AudioSegment.from_mp3(tts_path)
        tts_len = len(tts_audio)

        if tts_len <= slot_ms:
            final = final.overlay(tts_audio, position=start_ms)
        else:
            speed_factor = tts_len / slot_ms
            if speed_factor <= MAX_SPEED:
                sped = _speed_up(tts_audio, speed_factor, tmp_dir)
                final = final.overlay(sped[:slot_ms], position=start_ms)
            else:
                sped = _speed_up(tts_audio, MAX_SPEED, tmp_dir)
                final = final.overlay(sped[:slot_ms], position=start_ms)

        pct = int((i + 1) / total * 100)
        _update(progress_cb, "stitching", pct)

    return final


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(youtube_url: str, job_id: str, progress_cb: Callable | None = None) -> str:
    """
    Execute the full dubbing pipeline.
    Returns path to the final MP3 file.
    """
    with tempfile.TemporaryDirectory(prefix="dubber_") as tmp_dir:
        # Step 1: Download
        _update(progress_cb, "downloading", 0)
        mp3_path = download_audio(youtube_url, tmp_dir)
        _update(progress_cb, "downloading", 100)

        # Step 2: Transcribe
        _update(progress_cb, "transcribing", 0)
        segments = transcribe_audio(mp3_path)
        _update(progress_cb, "transcribing", 100)

        # Step 3: Translate
        _update(progress_cb, "translating", 0)
        segments = translate_segments(segments, progress_cb)

        # Step 4: TTS
        _update(progress_cb, "generating_tts", 0)
        segments = generate_tts(segments, tmp_dir, progress_cb)

        # Step 5: Stitch
        _update(progress_cb, "stitching", 0)
        final_audio = stitch_audio(segments, tmp_dir, progress_cb)

        # Step 6: Export
        _update(progress_cb, "exporting", 0)
        output_path = str(OUTPUT_DIR / f"{job_id}.mp3")
        final_audio.export(output_path, format="mp3", bitrate="128k")
        _update(progress_cb, "exporting", 100)

    return output_path

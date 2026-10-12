import base64
import os
import subprocess
import tempfile
from pathlib import Path

import requests


def _duration_seconds(video_path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def ingest_video(video_bytes, source_name, model="qwen2.5vl:3b", ollama_url="http://localhost:11434", interval=30, max_frames=12):
    """Sample lecture-video frames and transcribe visible slide/board content with timestamps."""
    with tempfile.TemporaryDirectory() as td:
        video_path = Path(td) / source_name
        video_path.write_bytes(video_bytes)
        duration = _duration_seconds(video_path)
        count = min(max_frames, max(1, int(duration // interval) + 1))
        records = []
        for i in range(count):
            timestamp = min(i * interval, max(0.0, duration - 0.1))
            frame = Path(td) / f"frame_{i:03d}.jpg"
            subprocess.run(
                ["ffmpeg", "-y", "-ss", str(timestamp), "-i", str(video_path), "-frames:v", "1", "-vf", "scale=1280:-1", str(frame)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
            )
            image_b64 = base64.b64encode(frame.read_bytes()).decode("ascii")
            prompt = (
                "Transcribe the educational content visible in this lecture frame. "
                "Capture slide text, equations, diagram labels, and board content. "
                "Do not invent missing content. Return concise study notes only."
            )
            response = requests.post(
                f"{ollama_url.rstrip('/')}/api/chat",
                json={"model": model, "stream": False, "messages":[{"role":"user","content":prompt,"images":[image_b64]}], "options":{"temperature":0}},
                timeout=150,
            )
            response.raise_for_status()
            text = response.json().get("message", {}).get("content", "").strip()
            records.append({
                "page": f"timestamp {timestamp:.1f}s",
                "source": source_name,
                "text": text,
                "raw_text": text,
                "vision_text": text,
                "extraction_method": "video_frame_vision",
                "recovery_status": "success",
                "timestamp": timestamp,
            })
        return records

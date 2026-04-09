import asyncio
import json
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import AsyncGenerator, List

import anthropic
import yt_dlp
from faster_whisper import WhisperModel
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

app = FastAPI()
executor = ThreadPoolExecutor(max_workers=1)

_whisper_model = None
_anthropic_client = None


def get_whisper() -> WhisperModel:
    global _whisper_model
    if _whisper_model is None:
        print("Loading Whisper model (first run only)...")
        _whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
    return _whisper_model


def get_anthropic() -> anthropic.Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic()
    return _anthropic_client


class SummarizeRequest(BaseModel):
    urls: List[str]


def _download_video(url: str, output_dir: str) -> str:
    output_template = os.path.join(output_dir, "video.%(ext)s")
    ydl_opts = {
        "outtmpl": output_template,
        "format": "best",
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    files = list(Path(output_dir).glob("video.*"))
    if not files:
        raise RuntimeError("Download failed — no file produced.")
    return str(files[0])


def _extract_audio(video_path: str, audio_path: str) -> None:
    result = subprocess.run(
        [
            "ffmpeg", "-i", video_path,
            "-ar", "16000", "-ac", "1",
            "-c:a", "pcm_s16le",
            audio_path, "-y", "-loglevel", "quiet",
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()}")


def _transcribe(audio_path: str) -> str:
    model = get_whisper()
    segments, _ = model.transcribe(audio_path, beam_size=5)
    return " ".join(seg.text.strip() for seg in segments).strip()


def _summarize(transcript: str) -> dict:
    if not transcript:
        return {
            "title": "No speech detected",
            "summary": "This video contains no detectable spoken content — it may be music, silent, or text-only.",
            "key_points": [],
        }

    client = get_anthropic()
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": f"""Summarize this TikTok video transcript into structured key points.

Transcript:
{transcript}

Respond ONLY with valid JSON in this exact format:
{{
  "title": "Short descriptive title (max 8 words)",
  "summary": "2-3 sentence overview of what this video is about",
  "key_points": ["point 1", "point 2", "point 3"]
}}

Include 3-5 key points. Be concise and informative.""",
            }
        ],
    )

    text = message.content[0].text.strip()

    # Strip markdown code fences if Claude wraps the JSON
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1].lstrip("json").strip() if len(parts) > 1 else text

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"title": "Summary", "summary": text, "key_points": []}


async def _stream_results(urls: List[str]) -> AsyncGenerator[str, None]:
    loop = asyncio.get_event_loop()

    for i, raw_url in enumerate(urls):
        url = raw_url.strip()
        if not url:
            continue

        yield f"data: {json.dumps({'type': 'start', 'url': url, 'index': i})}\n\n"

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                yield f"data: {json.dumps({'type': 'progress', 'url': url, 'index': i, 'step': 'Downloading video...'})}\n\n"
                video_path = await loop.run_in_executor(executor, _download_video, url, tmpdir)

                yield f"data: {json.dumps({'type': 'progress', 'url': url, 'index': i, 'step': 'Extracting audio...'})}\n\n"
                audio_path = os.path.join(tmpdir, "audio.wav")
                await loop.run_in_executor(executor, _extract_audio, video_path, audio_path)

                yield f"data: {json.dumps({'type': 'progress', 'url': url, 'index': i, 'step': 'Transcribing speech...'})}\n\n"
                transcript = await loop.run_in_executor(executor, _transcribe, audio_path)

                yield f"data: {json.dumps({'type': 'progress', 'url': url, 'index': i, 'step': 'Generating summary...'})}\n\n"
                summary = await loop.run_in_executor(executor, _summarize, transcript)

                yield f"data: {json.dumps({'type': 'result', 'url': url, 'index': i, 'data': summary})}\n\n"

        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'url': url, 'index': i, 'message': str(exc)})}\n\n"

    yield f"data: {json.dumps({'type': 'done'})}\n\n"


@app.post("/summarize")
async def summarize(request: SummarizeRequest):
    return StreamingResponse(
        _stream_results(request.urls),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/", response_class=HTMLResponse)
async def root():
    return (Path(__file__).parent / "index.html").read_text()

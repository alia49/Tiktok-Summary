# TikTok Summarizer

A local web app that takes TikTok video links and generates AI-powered summaries with key points for each video.

## How it works

1. Paste one or more TikTok URLs (comma-separated or one per line)
2. Each video is downloaded, audio is extracted and transcribed, then summarized by Claude AI
3. Results appear as cards in real time as each video is processed

## Tech stack

- **Backend:** Python + FastAPI
- **Downloader:** yt-dlp
- **Transcription:** faster-whisper (runs locally)
- **Summarization:** Claude API (claude-sonnet-4-6)
- **Frontend:** Vanilla HTML/CSS/JS with Server-Sent Events for live updates

## Prerequisites

- Python 3.10+
- ffmpeg (`brew install ffmpeg`)
- An [Anthropic API key](https://console.anthropic.com)

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set your Anthropic API key
export ANTHROPIC_API_KEY=your-key-here
```

## Run

```bash
uvicorn server:app --reload
```

Then open [http://localhost:8000](http://localhost:8000) in your browser.

> **Note:** The first run will download the Whisper `base` model (~150MB) automatically.

## Notes

- Works with public TikTok videos only
- Processing takes ~30–60 seconds per video depending on length
- Videos with no speech will return a "no speech detected" result

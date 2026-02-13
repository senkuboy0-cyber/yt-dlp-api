# Universal Video Downloader API

A robust, production-ready REST API built with FastAPI and yt-dlp to extract video metadata and formats from hundreds of sites involved in social media (YouTube, TikTok, Instagram, Facebook, X/Twitter, etc.).

## Features
- **Universal Support**: Works with any URL supported by yt-dlp.
- **Auto-Updating**: Automatically updates `yt-dlp` every 24 hours to stay ahead of platform changes.
- **Structured JSON**: Returns clean, categorized data (Video+Audio, Video Only, Audio Only).
- **Dockerized**: Ready for production deployment with `ffmpeg` included.

## Quick Start

### 1. Build the Docker Image
```bash
docker build -t video-api .
```

### 2. Run the Container
```bash
docker run -d -p 8000:8000 --name video-api video-api
```

### 3. Test the API
**Health Check:**
```bash
curl http://localhost:8000/ping
# Output: {"status": "alive"}
```

**Get Video Info:**
```bash
curl "http://localhost:8000/api/getinfo?url=https://www.youtube.com/watch?v=dQw4w9WgXcQ"
```

## API Response Structure
The API returns a JSON object with:
- `id`, `title`, `description`, `uploader`, `duration`, `thumbnail`, `platform`
- `video_with_audio`: List of streams containing both video and audio.
- `video_only`: List of video-only streams (deduplicated by resolution).
- `audio_only`: List of audio-only streams (sorted by bitrate).

## Auto-Update Mechanism
 The container includes a scheduled task that runs every 24 hours to execute:
`pip install -U yt-dlp`
This ensures the extractor logic is always up-to-date. You can see the update logs in the container output:
```bash
docker logs video-api
```

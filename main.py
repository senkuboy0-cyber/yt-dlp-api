import logging
import asyncio
import subprocess
import sys
import os
import re
from typing import List, Optional, Any, Dict
from datetime import datetime

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
import yt_dlp

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Universal Video Downloader API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VideoFormat(BaseModel):
    format_id: str
    ext: str
    url: Optional[str] = None
    resolution: Optional[str] = None
    filesize: Optional[int] = None
    note: Optional[str] = None
    vcodec: Optional[str] = None
    acodec: Optional[str] = None
    fps: Optional[float] = None
    bitrate: Optional[float] = None

class VideoMetadata(BaseModel):
    id: str
    title: str
    description: str
    uploader: str
    duration: int
    thumbnail: str
    platform: str
    view_count: Optional[int] = None
    upload_date: Optional[str] = None
    video_with_audio: List[VideoFormat]
    video_only: List[VideoFormat]
    audio_only: List[VideoFormat]


def run_update_ytdlp():
    logger.info("Starting scheduled yt-dlp update...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-U", "yt-dlp"],
            capture_output=True,
            text=True,
            check=True
        )
        logger.info(f"yt-dlp update successful: {result.stdout}")
    except subprocess.CalledProcessError as e:
        logger.error(f"yt-dlp update failed: {e.stderr}")
    except Exception as e:
        logger.error(f"Unexpected error during yt-dlp update: {str(e)}")


def resolution_rank(resolution: Optional[str]) -> int:
    if not resolution:
        return 0

    match = re.search(r"(\d+)", resolution)
    return int(match.group(1)) if match else 0


def process_formats(formats: List[Dict[str, Any]]) -> Dict[str, List[VideoFormat]]:
    video_with_audio = []
    video_only_best = {}
    audio_only = []

    for f in formats:
        download_url = f.get('url') or f.get('manifest_url')

        if not download_url:
            continue

        fmt_id = f.get('format_id')
        ext = f.get('ext')
        resolution = f.get('resolution') or f"{f.get('width', '?')}x{f.get('height', '?')}"
        filesize = f.get('filesize') or f.get('filesize_approx')
        note = f.get('format_note', '')

        vcodec_val = f.get('vcodec')
        acodec_val = f.get('acodec')
        vcodec = str(vcodec_val) if vcodec_val is not None else 'none'
        acodec = str(acodec_val) if acodec_val is not None else 'none'

        fps = f.get('fps')
        tbr = f.get('tbr') or f.get('abr')

        vf = VideoFormat(
            format_id=fmt_id,
            ext=ext,
            url=download_url,
            resolution=resolution,
            filesize=filesize,
            note=note,
            vcodec=vcodec,
            acodec=acodec,
            fps=fps,
            bitrate=tbr
        )

        is_video = vcodec != 'none'
        is_audio = acodec != 'none'

        if is_video and is_audio:
            video_with_audio.append(vf)

        elif is_video:
            if resolution not in video_only_best:
                video_only_best[resolution] = vf
            else:
                current_best = video_only_best[resolution]

                new_bitrate = tbr or 0
                curr_bitrate = current_best.bitrate or 0

                if new_bitrate > curr_bitrate:
                    video_only_best[resolution] = vf
                elif new_bitrate == curr_bitrate:
                    new_size = filesize or 0
                    curr_size = current_best.filesize or 0

                    if new_size > curr_size:
                        video_only_best[resolution] = vf

        elif is_audio:
            audio_only.append(vf)

    video_only = list(video_only_best.values())

    video_with_audio.sort(
        key=lambda x: (resolution_rank(x.resolution), x.bitrate or 0),
        reverse=True
    )

    video_only.sort(
        key=lambda x: (resolution_rank(x.resolution), x.bitrate or 0),
        reverse=True
    )

    audio_only.sort(key=lambda x: x.bitrate or 0, reverse=True)

    return {
        "video_with_audio": video_with_audio,
        "video_only": video_only,
        "audio_only": audio_only
    }


def get_highest_res_thumbnail(thumbnails: List[Dict[str, Any]]) -> str:
    if not thumbnails:
        return ""

    best = thumbnails[-1]

    for t in thumbnails:
        current_h = t.get('height', 0) or 0
        best_h = best.get('height', 0) or 0

        if current_h > best_h:
            best = t

    return best.get('url', '')


scheduler = AsyncIOScheduler()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    scheduler.add_job(
        run_update_ytdlp,
        IntervalTrigger(hours=24),
        id="update_ytdlp_job",
        replace_existing=True,
        next_run_time=datetime.now()
    )
    scheduler.start()
    logger.info("Scheduler started. yt-dlp auto-update configured.")

    yield

    # Shutdown
    scheduler.shutdown()
    logger.info("Scheduler shut down.")

app.router.lifespan_context = lifespan


@app.get("/ping")
@app.head("/ping")
async def ping():
    return {"status": "alive"}


@app.get("/api/getinfo", response_model=VideoMetadata)
async def get_info(url: str = Query(..., description="The URL of the video to extract info from")):
    try:
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
        }

        loop = asyncio.get_event_loop()
        info_dict = await loop.run_in_executor(None, lambda: _extract(url, ydl_opts))

        formats_categorized = process_formats(info_dict.get('formats', []))

        thumbnail = get_highest_res_thumbnail(info_dict.get('thumbnails', []))

        if not thumbnail:
            thumbnail = info_dict.get('thumbnail', '')

        return VideoMetadata(
            id=info_dict.get('id') or 'unknown',
            title=info_dict.get('title') or 'Unknown Title',
            description=info_dict.get('description') or '',
            uploader=info_dict.get('uploader') or 'Unknown Uploader',
            duration=int(info_dict.get('duration') or 0),
            thumbnail=thumbnail,
            platform=info_dict.get('extractor') or 'unknown',
            view_count=int(info_dict.get('view_count') or 0) if info_dict.get('view_count') is not None else None,
            upload_date=info_dict.get('upload_date'),
            video_with_audio=formats_categorized['video_with_audio'],
            video_only=formats_categorized['video_only'],
            audio_only=formats_categorized['audio_only']
        )

    except yt_dlp.utils.DownloadError as e:
        logger.error(f"DownloadError for URL {url}: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid URL or content unavailable: {str(e)}")

    except Exception as e:
        logger.error(f"Extraction failed for URL {url}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")


def _extract(url: str, opts: dict):
    if os.path.exists('cookies.txt'):
        opts['cookiefile'] = 'cookies.txt'

    with yt_dlp.YoutubeDL(opts) as ydl:
        return ydl.extract_info(url, download=False)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

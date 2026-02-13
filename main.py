import logging
import asyncio
import subprocess
import sys
from typing import List, Optional, Any, Dict
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
import yt_dlp

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Universal Video Downloader API")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Data Models ---

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
    bitrate: Optional[float] = None # tbr or abr

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

# --- Helper Functions ---

def run_update_ytdlp():
    """Runs the pip install command to upgrade yt-dlp."""
    logger.info("Starting scheduled yt-dlp update...")
    try:
        # Using sys.executable ensures we use the same python environment
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

def process_formats(formats: List[Dict[str, Any]]) -> Dict[str, List[VideoFormat]]:
    """Categorizes formats into video_with_audio, video_only, and audio_only."""
    video_with_audio = []
    video_only = []
    audio_only = []
    video_only_best = {}

    for f in formats:
        # ডাউনলোড লিঙ্ক (URL) সংগ্রহ করা
        download_url = f.get('url') or f.get('manifest_url')
        
        # যদি লিঙ্ক না থাকে, তবে এই ফরম্যাটটি স্কিপ করা হবে
        if not download_url:
            continue

        fmt_id = f.get('format_id')
        ext = f.get('ext')
        resolution = f.get('resolution') or f"{f.get('width', '?')}x{f.get('height', '?')}"
        filesize = f.get('filesize') or f.get('filesize_approx')
        note = f.get('format_note', '')
        vcodec = f.get('vcodec', 'none')
        acodec = f.get('acodec', 'none')
        fps = f.get('fps')
        tbr = f.get('tbr') or f.get('abr')

        # VideoFormat অবজেক্ট তৈরি (এখানে url=download_url যোগ করা হয়েছে)
        vf = VideoFormat(
            format_id=fmt_id,
            ext=ext,
            url=download_url, # এটিই বাটন কাজ না করার প্রধান কারণ ছিল
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
            # Logic for keeping best quality video_only for each resolution
            # We use tbr (total bitrate) as a proxy for quality if available, else filesize
            if resolution not in video_only_best:
                video_only_best[resolution] = vf
            else:
                current_best = video_only_best[resolution]
                # Compare bitrates
                new_bitrate = tbr or 0
                curr_bitrate = current_best.bitrate or 0
                if new_bitrate > curr_bitrate:
                    video_only_best[resolution] = vf
                elif new_bitrate == curr_bitrate:
                    # Fallback to filesize if bitrates are equal or missing
                    new_size = filesize or 0
                    curr_size = current_best.filesize or 0
                    if new_size > curr_size:
                        video_only_best[resolution] = vf

        elif is_audio:
            audio_only.append(vf)

    # Convert deduplicated dictionary back to list
    video_only = list(video_only_best.values())

    # Sort logic
    # Video w/ Audio: highest resolution/bitrate first
    video_with_audio.sort(key=lambda x: (x.resolution, x.bitrate or 0), reverse=True)
    # Video Only: highest resolution first (already deduped for best quality per res)
    video_only.sort(key=lambda x: (x.resolution, x.bitrate or 0), reverse=True)
    # Audio Only: highest bitrate first
    audio_only.sort(key=lambda x: x.bitrate or 0, reverse=True)

    return {
        "video_with_audio": video_with_audio,
        "video_only": video_only,
        "audio_only": audio_only
    }

def get_highest_res_thumbnail(thumbnails: List[Dict[str, Any]]) -> str:
    """Finds the thumbnail with the highest preference/resolution."""
    if not thumbnails:
        return ""
    # yt-dlp usually sorts thumbnails by quality, last is best. 
    # But let's be safe and look for 'preference' or 'height'
    best = thumbnails[-1]
    for t in thumbnails:
        current_h = t.get('height', 0) or 0
        best_h = best.get('height', 0) or 0
        if current_h > best_h:
            best = t
    return best.get('url', '')

# --- Background Scheduler ---

scheduler = AsyncIOScheduler()

@app.on_event("startup")
async def start_scheduler():
    # Schedule yt-dlp update every 24 hours
    scheduler.add_job(
        run_update_ytdlp,
        IntervalTrigger(hours=24),
        id="update_ytdlp_job",
        replace_existing=True,
        next_run_time=datetime.now() # Run immediately on startup to ensure latest version
    )
    scheduler.start()
    logger.info("Scheduler started. yt-dlp auto-update configured.")

@app.on_event("shutdown")
async def shutdown_scheduler():
    scheduler.shutdown()
    logger.info("Scheduler shut down.")

# --- Endpoints ---

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
            'extract_flat': False, # We need full info
            # We don't want to download, just get info
        }

        # Run extraction in a separate thread to prevent blocking the event loop
        # We use a wrapper function because YoutubeDL context manager needs to happen inside the thread?
        # Actually it's safer to instantiate inside the thread or use blocking call directly
        
        loop = asyncio.get_event_loop()
        info_dict = await loop.run_in_executor(None, lambda: _extract(url, ydl_opts))

        # Process metadata
        formats_categorized = process_formats(info_dict.get('formats', []))
        
        # Get highest res thumbnail
        thumbnail = get_highest_res_thumbnail(info_dict.get('thumbnails', []))
        if not thumbnail:
            thumbnail = info_dict.get('thumbnail', '')

        return VideoMetadata(
            id=info_dict.get('id', 'unknown'),
            title=info_dict.get('title', 'Unknown Title'),
            description=info_dict.get('description', ''),
            uploader=info_dict.get('uploader', 'Unknown Uploader'),
            duration=info_dict.get('duration', 0) or 0,
            thumbnail=thumbnail,
            platform=info_dict.get('extractor', 'unknown'),
            view_count=info_dict.get('view_count'),
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
    opts['cookiefile'] = 'cookies.txt' 
    with yt_dlp.YoutubeDL(opts) as ydl:
        return ydl.extract_info(url, download=False)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

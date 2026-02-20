import os
import time
import glob
import random
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Request, Depends, Query, Security
from fastapi.responses import ORJSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel

import yt_dlp

os.environ["TZ"] = "UTC"
API_KEY = os.getenv("TITAN_SECRET_KEY", "Titan_2026_Ultra_Fast")
MAX_WORKERS = int(os.getenv("MAX_THREADS", "100")) 
CACHE_TTL = 14400  

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("TitanAPI")

class FormatModel(BaseModel):
    format_id: str
    ext: str
    resolution: Optional[str] = "audio only"
    vcodec: str
    acodec: str
    url: str

class ThumbnailModel(BaseModel):
    url: str
    width: int
    height: int

class MediaResponse(BaseModel):
    success: bool = True
    process_time_ms: float
    cached: bool = False
    video_id: str
    title: str
    duration: int
    is_live: bool
    thumbnails: List[ThumbnailModel] = []
    direct_stream_url: str
    fallback_streams: List[FormatModel] = []

class ErrorResponse(BaseModel):
    success: bool = False
    error_code: int
    message: str
    process_time_ms: float

class EnterpriseCookieManager:
    def __init__(self, directory: str = "cookies"):
        self.directory = directory
        self.pool: List[str] = []
        self.banned: Dict[str, float] = {}
        self.ban_time = 3600  
        self._ensure_dir()
        self.refresh_pool()

    def _ensure_dir(self):
        if not os.path.exists(self.directory):
            os.makedirs(self.directory, exist_ok=True)

    def refresh_pool(self):
        files = glob.glob(os.path.join(self.directory, "*.txt"))
        now = time.time()
        
        for cookie in list(self.banned.keys()):
            if now - self.banned[cookie] > self.ban_time:
                del self.banned[cookie]

        self.pool = [f for f in files if f not in self.banned and os.path.getsize(f) > 0]
        logger.info(f"Cookie Pool: Active: {len(self.pool)}, Banned: {len(self.banned)}")

    def get_cookie(self) -> Optional[str]:
        if not self.pool:
            self.refresh_pool()
        if not self.pool:
            return None
        return random.choice(self.pool)

    def report_failure(self, cookie_path: str):
        if cookie_path in self.pool:
            self.banned[cookie_path] = time.time()
            self.pool.remove(cookie_path)
            logger.warning(f"Cookie Banned (Temp): {cookie_path}")

cookie_vault = EnterpriseCookieManager()

class MemoryCache:
    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            if key in self._cache:
                item = self._cache[key]
                if time.time() - item['timestamp'] < CACHE_TTL:
                    return item['data']
                else:
                    del self._cache[key]
        return None

    async def set(self, key: str, data: Dict[str, Any]):
        async with self._lock:
            self._cache[key] = {'timestamp': time.time(), 'data': data}

    async def cleanup(self):
        async with self._lock:
            now = time.time()
            expired = [k for k, v in self._cache.items() if now - v['timestamp'] >= CACHE_TTL]
            for k in expired: del self._cache[k]

cache_engine = MemoryCache()

class YtDlpLogger:
    def debug(self, msg): pass
    def warning(self, msg): pass
    def error(self, msg): logger.error(f"yt-dlp Core Error: {msg}")

class TitanExtractor:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="Titan")

    def _build_config(self, audio_only: bool, cookie_path: Optional[str] = None) -> dict:
        config = {
            'quiet': True,
            'no_warnings': True,
            'skip_download': True,
            'extract_flat': False,
            'noplaylist': True,
            'logger': YtDlpLogger(),
            
            'check_formats': True,  
            'youtube_include_dash_manifest': False,
            'youtube_include_hls_manifest': False,
            
            'format': 'bestaudio/best' if audio_only else 'best',
            'nocheckcertificate': True,
            'extractor_args': {
                'youtube': {
                    'player_client': ['android', 'ios', 'web'],
                    'player_skip': ['js', 'configs', 'webpage']
                }
            }
        }
        if cookie_path and os.path.exists(cookie_path):
            config['cookiefile'] = cookie_path
        return config

    def _sync_extract(self, url: str, audio_only: bool, retries: int = 3) -> dict:
        last_err = None
        for attempt in range(retries):
            cookie = cookie_vault.get_cookie()
            config = self._build_config(audio_only, cookie)
            
            try:
                with yt_dlp.YoutubeDL(config) as ydl:
                    info = ydl.extract_info(url, download=False)
                    if info:
                        return info
            except Exception as e:
                last_err = str(e)
                logger.warning(f"Attempt {attempt+1}/{retries} failed using cookie {cookie}: {last_err}")
                if cookie:
                    cookie_vault.report_failure(cookie)
                time.sleep(1) 
                
        raise Exception(f"يوتيوب يرفض الاتصال بعد تجربة عدة حسابات. السبب: {last_err}")

    async def extract_smart(self, url: str, audio_only: bool) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, self._sync_extract, url, audio_only, 3)

titan_core = TitanExtractor()

app = FastAPI(
    title="TitanOS Enterprise Media API",
    version="5.1.0",
    default_response_class=ORJSONResponse,
    docs_url=None,
    redoc_url=None
)

app.add_middleware(GZipMiddleware, minimum_size=500)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key_header = APIKeyHeader(name="X-Titan-Key", auto_error=False)

async def verify_auth(key: str = Security(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized Access to TitanOS Core")
    return key

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(background_tasks())

async def background_tasks():
    while True:
        await asyncio.sleep(600) 
        await cache_engine.cleanup()
        cookie_vault.refresh_pool()

@app.get("/", tags=["System"])
async def index():
    return ORJSONResponse({"system": "TitanOS Core", "status": "Operational", "mode": "Turbo Verified"})

@app.get("/api/v1/extract", response_model=MediaResponse, tags=["Media"])
async def extract_media(
    url: str = Query(..., description="Target YouTube URL"),
    audio_only: bool = Query(True, description="Optimize for audio streams"),
    force_refresh: bool = Query(False, description="Bypass cache"),
    auth: str = Depends(verify_auth)
):
    t0 = time.perf_counter()
    cache_key = f"{url}_audio:{audio_only}"

    try:
        if not force_refresh:
            cached_data = await cache_engine.get(cache_key)
            if cached_data:
                cached_data['cached'] = True
                cached_data['process_time_ms'] = round((time.perf_counter() - t0) * 1000, 3)
                return ORJSONResponse(cached_data)

        info = await titan_core.extract_smart(url, audio_only)

        direct_url = info.get("url")
        fallback_list = []
        
        for f in info.get("formats", []):
            protocol = str(f.get("protocol", "")).lower()
            if protocol.startswith(("http", "m3u8")):
                fmt_obj = FormatModel(
                    format_id=str(f.get("format_id", "0")),
                    ext=f.get("ext", "unknown"),
                    resolution=f.get("format_note") or f.get("resolution") or "audio",
                    vcodec=f.get("vcodec", "none"),
                    acodec=f.get("acodec", "none"),
                    url=f.get("url", "")
                )
                fallback_list.append(fmt_obj)
                
                if not direct_url:
                    if audio_only and fmt_obj.vcodec == "none" and fmt_obj.acodec != "none":
                        direct_url = fmt_obj.url
                    elif not audio_only and fmt_obj.vcodec != "none":
                        direct_url = fmt_obj.url

        if not direct_url and fallback_list:
            direct_url = fallback_list[-1].url 

        if not direct_url:
            raise ValueError("No playable stream found.")

        thumbs = [ThumbnailModel(url=t.get("url"), width=t.get("width", 0), height=t.get("height", 0)) for t in info.get("thumbnails", [])]

        response_data = {
            "success": True,
            "process_time_ms": round((time.perf_counter() - t0) * 1000, 3),
            "cached": False,
            "video_id": info.get("id", ""),
            "title": info.get("title", "Unknown"),
            "duration": info.get("duration") or 0,
            "is_live": info.get("is_live") or info.get("was_live") or False,
            "thumbnails": [t.model_dump() for t in thumbs[-1:]], 
            "direct_stream_url": direct_url,
            "fallback_streams": [f.model_dump() for f in fallback_list[-3:]] 
        }

        await cache_engine.set(cache_key, response_data)
        return ORJSONResponse(response_data)

    except Exception as e:
        logger.error(f"Extraction Error for {url}: {str(e)}")
        return ORJSONResponse(
            status_code=500,
            content=ErrorResponse(
                success=False,
                error_code=500,
                message=str(e),
                process_time_ms=round((time.perf_counter() - t0) * 1000, 3)
            ).model_dump()
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, loop="uvloop", http="httptools", workers=4)

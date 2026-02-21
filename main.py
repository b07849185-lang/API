import os
import time
import glob
import random
import asyncio
import logging
import contextlib
import urllib.parse
import re
import traceback
import sys
import secrets as _secrets
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Dict, Any, Tuple
from fastapi import FastAPI, HTTPException, Depends, Query, Security
from fastapi.responses import ORJSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
import orjson
import yt_dlp

os.environ.setdefault("TZ", "UTC")
API_KEY = os.getenv("TITAN_SECRET_KEY", "Titan_2026_Ultra_Fast")
COOKIE_DIR = os.getenv("TITAN_COOKIE_DIR", "cookies")
MAX_CONCURRENT_EXTRACTS = int(os.getenv("MAX_THREADS", "32"))  
CACHE_TTL = int(os.getenv("CACHE_TTL", "14400"))
COOKIE_BAN_TIME = int(os.getenv("COOKIE_BAN_TIME", "3600"))
IMPERSONATE_TARGET = os.getenv("YT_IMPERSONATE_TARGET", "chrome110")
YT_DLP_BIN = os.getenv("YT_DLP_BIN", "yt-dlp")

try:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("TitanAPI")
    logger.setLevel(logging.INFO)
except Exception:
    logger = None

class StreamInfo(BaseModel):
    format_id: str
    ext: str
    resolution: str
    vcodec: str
    acodec: str
    url: str
    quality_score: int

class SmartFormats(BaseModel):
    best_muxed: List[StreamInfo] = []
    audio_only: List[StreamInfo] = []
    video_only: List[StreamInfo] = []

class ThumbnailModel(BaseModel):
    url: str
    width: int = 0
    height: int = 0

class MediaResponse(BaseModel):
    success: bool = True
    process_time_ms: float
    cached: bool = False
    extraction_method: str
    video_id: str
    title: str
    duration: int
    is_live: bool
    thumbnails: List[ThumbnailModel] = []
    direct_stream_url: str
    smart_formats: SmartFormats
    raw_fallback_count: int

class ErrorResponse(BaseModel):
    success: bool = False
    error_code: int
    message: str
    process_time_ms: float

def _loads_bytes(b: bytes) -> dict:
    try:
        return orjson.loads(b)
    except Exception:
        return {}

def dict_to_model(m):
    try:
        if hasattr(m, "model_dump"):
            return m.model_dump()
        if hasattr(m, "dict"):
            return m.dict()
        return m
    except Exception:
        return {}

class EnterpriseCookieManager:
    def __init__(self, directory: str = COOKIE_DIR, ban_time: int = COOKIE_BAN_TIME, reuse_delay: int = 2):
        self.directory = directory
        self.pool: List[str] = []
        self.banned: Dict[str, float] = {}
        self.last_used: Dict[str, float] = {}
        self.ban_time = ban_time
        self.reuse_delay = reuse_delay
        self._lock = asyncio.Lock()
        try:
            os.makedirs(self.directory, exist_ok=True)
        except Exception:
            pass
        self.refresh_pool_sync()

    def refresh_pool_sync(self):
        try:
            files = glob.glob(os.path.join(self.directory, "*.txt"))
            now = time.time()
            for c in list(self.banned.keys()):
                try:
                    if now - self.banned[c] > self.ban_time:
                        del self.banned[c]
                except Exception:
                    continue
            self.pool = [f for f in files if f not in self.banned and os.path.getsize(f) > 0]
        except Exception:
            self.pool = []

    async def get_cookie(self) -> Optional[str]:
        try:
            async with self._lock:
                if not self.pool:
                    self.refresh_pool_sync()
                if not self.pool:
                    return None
                candidates = [c for c in self.pool if (time.time() - self.last_used.get(c, 0)) > self.reuse_delay]
                if not candidates:
                    candidates = self.pool
                if not candidates:
                    return None
                chosen = random.choice(candidates)
                self.last_used[chosen] = time.time()
                return chosen
        except Exception:
            return None

    async def report_failure(self, cookie_path: str):
        try:
            async with self._lock:
                self.banned[cookie_path] = time.time()
                if cookie_path in self.pool:
                    self.pool.remove(cookie_path)
        except Exception:
            pass

cookie_vault = EnterpriseCookieManager()

class MemoryCache:
    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        try:
            async with self._lock:
                item = self._cache.get(key)
                if not item: 
                    return None
                if time.time() - item.get("timestamp", 0) < CACHE_TTL: 
                    return item.get("data")
                try:
                    del self._cache[key]
                except Exception:
                    pass
                return None
        except Exception:
            return None

    async def set(self, key: str, data: Dict[str, Any]):
        try:
            async with self._lock:
                self._cache[key] = {"timestamp": time.time(), "data": data}
        except Exception:
            pass

    async def cleanup(self):
        try:
            async with self._lock:
                now = time.time()
                expired = [k for k, v in self._cache.items() if now - v.get("timestamp", 0) >= CACHE_TTL]
                for k in expired: 
                    try:
                        del self._cache[k]
                    except Exception:
                        pass
        except Exception:
            pass

cache_engine = MemoryCache()

class TitanExtractor:
    def __init__(self):
        self.thread_pool = ThreadPoolExecutor(max_workers=max(8, min(MAX_CONCURRENT_EXTRACTS, 128)))
        self.sema = asyncio.Semaphore(MAX_CONCURRENT_EXTRACTS)
        self.impersonate_supported = True 

    def _sync_api_extract(self, url: str, cookie: Optional[str], attempt: int) -> dict:
        try:
            opts = {
                "quiet": True, 
                "no_warnings": True, 
                "skip_download": True, 
                "extract_flat": False, 
                "noplaylist": True,
                "default_search": "ytsearch",
                "socket_timeout": 8,
                "compat_opts": ["no-youtube-unavailable-videos"]
            }
            if cookie: 
                opts["cookiefile"] = cookie
            if self.impersonate_supported: 
                opts["impersonate"] = IMPERSONATE_TARGET

            if attempt == 1:
                opts["remote_components"] = "ejs:github"
                opts["extractor_args"] = {"youtube": {"player_client": ["android", "web"], "player_skip": ["webpage", "configs"]}}
            elif attempt == 2:
                opts["extractor_args"] = {"youtube": {"player_client": ["android"], "player_skip": ["webpage", "configs"]}}
            else:
                opts["remote_components"] = "ejs:github"
                opts["extractor_args"] = {"youtube": {"player_client": ["web"]}}

            with yt_dlp.YoutubeDL(opts) as ydl:
                return ydl.extract_info(url, download=False)
        except Exception as e:
            raise e

    async def extract_smart(self, raw_query: str) -> Tuple[Dict[str, Any], str]:
        last_err = "unknown"
        try:
            query = str(raw_query).strip()
            if not query.lower().startswith(("http://", "https://", "ytsearch", "ytsearch1:")):
                query = f"ytsearch1:{query}"
        except Exception:
            query = raw_query

        async with self.sema:
            for attempt in range(1, 4):  
                cookie = await cookie_vault.get_cookie()
                method_used = ""
                cookie_used = False 
                try:
                    if attempt == 1:
                        method_used = "In-Memory API (Android/Web + EJS)"
                    elif attempt == 2:
                        method_used = "In-Memory API (Android Fast)"
                    elif attempt == 3:
                        method_used = "In-Memory API (Web Fallback + EJS)"

                    if cookie: 
                        cookie_used = True
                        
                    try:
                        loop = asyncio.get_running_loop()
                        info = await loop.run_in_executor(self.thread_pool, self._sync_api_extract, query, cookie, attempt)
                        try:
                            if info and "entries" in info and isinstance(info["entries"], list) and len(info["entries"]) > 0:
                                info = info["entries"][0]
                        except Exception:
                            pass
                        if info and info.get("formats"): 
                            return info, method_used
                    except Exception as inner_e:
                        last_err = str(inner_e)

                except Exception as e:
                    last_err = str(e)

                try:
                    if cookie_used and cookie and ("Sign in" in last_err or "bot" in last_err.lower()):
                        await cookie_vault.report_failure(cookie)
                except Exception:
                    pass

                try:
                    await asyncio.sleep(0.5)
                except Exception:
                    pass

        raise Exception(f"Extraction failed. Err: {last_err}")

try:
    titan_core = TitanExtractor()
except Exception:
    titan_core = None

app = FastAPI(title="TitanOS Enterprise", version="9.0.0", default_response_class=ORJSONResponse, docs_url=None, redoc_url=None)

try:
    app.add_middleware(GZipMiddleware, minimum_size=256)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
except Exception:
    pass

try:
    api_key_header = APIKeyHeader(name="X-Titan-Key", auto_error=False)
except Exception:
    api_key_header = None

def verify_auth_token(key: Optional[str]) -> str:
    try:
        if not key or not _secrets.compare_digest(str(key), str(API_KEY)):
            raise HTTPException(status_code=403, detail="Forbidden")
        return key
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=403, detail="Forbidden")

async def verify_auth(key: str = Security(api_key_header)):
    try:
        return verify_auth_token(key)
    except Exception:
        raise HTTPException(status_code=403, detail="Forbidden")

@app.on_event("startup")
async def startup_event():
    try:
        app.state._bg_task = asyncio.create_task(_background_tasks())
    except Exception:
        pass

async def _background_tasks():
    try:
        while True:
            try:
                await asyncio.sleep(120)
                await cache_engine.cleanup()
                cookie_vault.refresh_pool_sync()
            except Exception:
                await asyncio.sleep(60)
    except asyncio.CancelledError:
        pass
    except Exception:
        pass

@app.on_event("shutdown")
async def shutdown_event():
    try:
        task = getattr(app.state, "_bg_task", None)
        if task:
            task.cancel()
    except Exception:
        pass

@app.get("/")
async def index():
    try:
        if os.path.exists("index.html"):
            with open("index.html", "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
        return ORJSONResponse({"system": "TitanOS Core", "status": "Active", "engine": "Zero Latency Engine"})
    except Exception:
        return ORJSONResponse({"status": "Active"})

def build_smart_formats(formats_list: List[dict]) -> Tuple[SmartFormats, str, str]:
    best_muxed, audio_only, video_only = [], [], []
    try:
        for f in formats_list:
            try:
                if not isinstance(f, dict): continue
                ext = str(f.get("ext", "")).lower()
                if ext in ["mhtml", "sb0", "sb1"]: continue
                proto = str(f.get("protocol", "")).lower()
                if not proto.startswith(("http", "https", "m3u8")): continue
                
                vcodec = str(f.get("vcodec", "none")).lower()
                acodec = str(f.get("acodec", "none")).lower()
                url = str(f.get("url", ""))
                if not url: continue
                
                fmt_id = str(f.get("format_id", "0"))
                res = str(f.get("format_note", f.get("resolution", "unknown")))
                tbr = float(f.get("tbr", 0.0) or 0.0)
                
                score = 0
                if "mp4" in ext or "m4a" in ext: score += 10
                score += int(tbr / 100)
                
                s_info = StreamInfo(
                    format_id=fmt_id, ext=ext, resolution=res,
                    vcodec=vcodec, acodec=acodec, url=url, quality_score=score
                )
                
                if vcodec != "none" and acodec != "none":
                    if "m3u8" not in proto: s_info.quality_score += 50
                    best_muxed.append(s_info)
                elif vcodec == "none" and acodec != "none":
                    audio_only.append(s_info)
                elif vcodec != "none" and acodec == "none":
                    video_only.append(s_info)
            except Exception:
                continue
                
        best_muxed.sort(key=lambda x: x.quality_score, reverse=True)
        audio_only.sort(key=lambda x: x.quality_score, reverse=True)
        video_only.sort(key=lambda x: x.quality_score, reverse=True)
        
        sf = SmartFormats(best_muxed=best_muxed, audio_only=audio_only, video_only=video_only)
        
        d_audio = audio_only[0].url if audio_only else ""
        d_video = best_muxed[0].url if best_muxed else (video_only[0].url if video_only else "")
        
        return sf, d_audio, d_video
    except Exception:
        return SmartFormats(), "", ""

@app.get("/api/v1/extract", response_model=MediaResponse)
async def extract_media(url: str = Query(...), audio_only: bool = Query(True), force_refresh: bool = Query(False), auth: str = Depends(verify_auth)):
    try:
        t0 = time.perf_counter()
    except Exception:
        t0 = 0
        
    try:
        cache_key = f"{url}_audio:{audio_only}"
    except Exception:
        cache_key = str(url)

    try:
        if not force_refresh:
            try:
                cached = await cache_engine.get(cache_key)
                if cached:
                    try:
                        cached['cached'] = True
                        cached['process_time_ms'] = round((time.perf_counter() - t0) * 1000, 3)
                    except Exception:
                        pass
                    return ORJSONResponse(cached)
            except Exception:
                pass

        info, method = await titan_core.extract_smart(url)

        try:
            raw_formats = info.get("formats", [])
            if not isinstance(raw_formats, list): raw_formats = []
        except Exception:
            raw_formats = []

        smart_formats, best_a, best_v = build_smart_formats(raw_formats)
        
        try:
            if audio_only:
                direct_url = best_a if best_a else best_v
            else:
                direct_url = best_v if best_v else best_a
            
            if not direct_url:
                direct_url = str(info.get("url", ""))
                
            if not direct_url and len(raw_formats) > 0:
                direct_url = str(raw_formats[-1].get("url", ""))
                
            if not direct_url: 
                raise ValueError("Extraction yielded no valid streams.")
        except Exception as ex:
            raise ValueError(f"Stream parsing failed: {ex}")

        try:
            raw_thumbs = info.get("thumbnails", [])
            if not isinstance(raw_thumbs, list): raw_thumbs = []
            thumbs = []
            for t in raw_thumbs:
                try:
                    thumbs.append(ThumbnailModel(url=str(t.get("url", "")), width=int(t.get("width", 0) or 0), height=int(t.get("height", 0) or 0)))
                except Exception:
                    continue
        except Exception:
            thumbs = []

        try:
            is_live = bool(info.get("is_live", False) or info.get("was_live", False))
        except Exception:
            is_live = False
            
        try:
            dur_raw = info.get("duration")
            dur = 0 if is_live or str(dur_raw).lower() in ["live", "none"] else int(dur_raw or 0)
        except Exception:
            dur = 0

        try:
            vid_id = str(info.get("id", ""))
            vid_title = str(info.get("title", "Unknown"))
        except Exception:
            vid_id = ""
            vid_title = "Unknown"

        try:
            response_data = {
                "success": True,
                "process_time_ms": round((time.perf_counter() - t0) * 1000, 3),
                "cached": False,
                "extraction_method": str(method),
                "video_id": vid_id,
                "title": vid_title,
                "duration": dur,
                "is_live": is_live,
                "thumbnails": [dict_to_model(t) for t in thumbs[-1:]] if thumbs else [],
                "direct_stream_url": str(direct_url),
                "smart_formats": dict_to_model(smart_formats),
                "raw_fallback_count": len(raw_formats)
            }
        except Exception:
            raise ValueError("Failed building final response structure")

        try:
            await cache_engine.set(cache_key, response_data)
        except Exception:
            pass
            
        return ORJSONResponse(response_data)

    except Exception as e:
        try:
            p_time = round((time.perf_counter() - t0) * 1000, 3)
        except Exception:
            p_time = 0.0
        try:
            return ORJSONResponse(status_code=500, content=dict_to_model(ErrorResponse(success=False, error_code=500, message=str(e), process_time_ms=p_time)))
        except Exception:
            return ORJSONResponse(status_code=500, content={"success": False, "error_code": 500, "message": "Critical Failure", "process_time_ms": 0.0})

if __name__ == "__main__":
    try:
        import uvicorn
        uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8080")), loop="uvloop", http="httptools", workers=int(os.getenv("UVICORN_WORKERS", "4")))
    except Exception:
        pass

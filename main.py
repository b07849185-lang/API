# main.py
"""
TitanOS - Hardened Extractor Service (6.6.0 - Speed Optimized)
Features:
- API key auth (constant-time compare)
- Enterprise Cookie pool with temporary bans & reuse-delay
- PROXY SYSTEM COMPLETELY REMOVED
- CLI Standard (EJS + Web) promoted to Attempt #1 for blazing fast response
"""
import os
import time
import glob
import random
import asyncio
import logging
import contextlib
import secrets as _secrets
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Dict, Any, Tuple

from fastapi import FastAPI, HTTPException, Depends, Query, Security
from fastapi.responses import ORJSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel

try:
    import orjson as _orjson  
    def _loads_bytes(b: bytes) -> dict:
        return _orjson.loads(b)
except Exception:
    import json as _json  
    def _loads_bytes(b: bytes) -> dict:
        return _json.loads(b.decode("utf-8", "ignore") if isinstance(b, bytes) else b)

import yt_dlp

# ============== Configuration ==============
os.environ.setdefault("TZ", "UTC")
API_KEY = os.getenv("TITAN_SECRET_KEY", "Titan_2026_Ultra_Fast")
COOKIE_DIR = os.getenv("TITAN_COOKIE_DIR", "cookies")
MAX_CONCURRENT_EXTRACTS = int(os.getenv("MAX_THREADS", "4"))  
CACHE_TTL = int(os.getenv("CACHE_TTL", "14400"))
COOKIE_BAN_TIME = int(os.getenv("COOKIE_BAN_TIME", "3600"))
IMPERSONATE_TARGET = os.getenv("YT_IMPERSONATE_TARGET", "chrome110")
YT_DLP_BIN = os.getenv("YT_DLP_BIN", "yt-dlp")  

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("TitanAPI")

# ============== Models ==============
class FormatModel(BaseModel):
    format_id: str
    ext: str
    resolution: Optional[str] = "audio only"
    vcodec: str
    acodec: str
    url: str

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
    fallback_streams: List[FormatModel] = []

class ErrorResponse(BaseModel):
    success: bool = False
    error_code: int
    message: str
    process_time_ms: float

def model_to_primitive(m):
    if hasattr(m, "model_dump"):
        return m.model_dump()
    return m.dict()

# ============== Enterprise Cookie Manager ==============
class EnterpriseCookieManager:
    def __init__(self, directory: str = COOKIE_DIR, ban_time: int = COOKIE_BAN_TIME, reuse_delay: int = 10):
        self.directory = directory
        self.pool: List[str] = []
        self.banned: Dict[str, float] = {}
        self.last_used: Dict[str, float] = {}
        self.ban_time = ban_time
        self.reuse_delay = reuse_delay
        self._lock = asyncio.Lock()
        os.makedirs(self.directory, exist_ok=True)
        self.refresh_pool_sync()

    def refresh_pool_sync(self):
        files = glob.glob(os.path.join(self.directory, "*.txt"))
        now = time.time()
        for c in list(self.banned.keys()):
            if now - self.banned[c] > self.ban_time:
                del self.banned[c]
        self.pool = [f for f in files if f not in self.banned and os.path.getsize(f) > 0]

    async def get_cookie(self) -> Optional[str]:
        async with self._lock:
            if not self.pool:
                self.refresh_pool_sync()
            if not self.pool:
                return None
            candidates = [c for c in self.pool if (time.time() - self.last_used.get(c, 0)) > self.reuse_delay]
            if not candidates:
                candidates = self.pool
            chosen = random.choice(candidates)
            self.last_used[chosen] = time.time()
            return chosen

    async def report_failure(self, cookie_path: str):
        async with self._lock:
            self.banned[cookie_path] = time.time()
            if cookie_path in self.pool:
                self.pool.remove(cookie_path)

cookie_vault = EnterpriseCookieManager()

# ============== Memory Cache ==============
class MemoryCache:
    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            item = self._cache.get(key)
            if not item: return None
            if time.time() - item["timestamp"] < CACHE_TTL: return item["data"]
            del self._cache[key]
            return None

    async def set(self, key: str, data: Dict[str, Any]):
        async with self._lock:
            self._cache[key] = {"timestamp": time.time(), "data": data}

    async def cleanup(self):
        async with self._lock:
            now = time.time()
            expired = [k for k, v in self._cache.items() if now - v["timestamp"] >= CACHE_TTL]
            for k in expired: del self._cache[k]

cache_engine = MemoryCache()

# ============== Titan Extractor ==============
class TitanExtractor:
    def __init__(self):
        self.thread_pool = ThreadPoolExecutor(max_workers=max(2, min(MAX_CONCURRENT_EXTRACTS, 32)))
        self.sema = asyncio.Semaphore(MAX_CONCURRENT_EXTRACTS)
        self.impersonate_supported = False
        try:
            out = self._run_cmd_sync([YT_DLP_BIN, "--list-impersonate-targets"])
            if out and ("chrome" in out or IMPERSONATE_TARGET in out):
                self.impersonate_supported = True
        except Exception:
            pass

    def _run_cmd_sync(self, cmd: List[str], timeout: int = 10) -> str:
        import subprocess
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        return p.stdout.decode("utf-8", "ignore")

    async def _exec_cmd(self, *args: str, timeout: int = 35) -> Tuple[bytes, bytes]:
        proc = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            return out or b"", err or b""
        except asyncio.TimeoutError:
            with contextlib.suppress(Exception): proc.kill()
            return b"", b"timeout"

    def _sync_api_extract(self, url: str, cookie: Optional[str]) -> dict:
        opts = {
            "quiet": True, "no_warnings": True, "skip_download": True, "extract_flat": False, "noplaylist": True,
            "extractor_args": {"youtube": {"player_client": ["web"]}},
        }
        if cookie: opts["cookiefile"] = cookie
        if self.impersonate_supported: opts["impersonate"] = IMPERSONATE_TARGET
        with yt_dlp.YoutubeDL(opts) as ydl:
            return ydl.extract_info(url, download=False)

    async def extract_smart(self, url: str, audio_only: bool) -> Tuple[Dict[str, Any], str]:
        last_err = "unknown"
        async with self.sema:
            for attempt in range(1, 4):  # Reduced to 3 highly targeted attempts
                cookie = await cookie_vault.get_cookie()
                method_used = ""
                cookie_used = False 

                try:
                    # 🚀 ATTEMPT 1: The Golden Key (CLI + EJS + Web + Cookie) - PROMOTED TO FIRST!
                    if attempt == 1:
                        method_used = "CLI Standard (Web Client + Remote EJS)"
                        cmd = [YT_DLP_BIN, "--dump-json", url, "--no-warnings", "--remote-components", "ejs:github"]
                        if cookie: 
                            cmd.extend(["--cookies", cookie])
                            cookie_used = True
                        if self.impersonate_supported: cmd.extend(["--impersonate", IMPERSONATE_TARGET])
                        cmd.extend(["--extractor-args", "youtube:player_client=web"])
                        out, err = await self._exec_cmd(*cmd, timeout=20)
                        if out:
                            info = _loads_bytes(out)
                            if info.get("formats"): return info, method_used
                        last_err = err.decode("utf-8", "ignore") if err else "Empty response"

                    # ATTEMPT 2: Fast API Fallback (Just in case)
                    elif attempt == 2:
                        method_used = "Fast API (Web Client + Cookie)"
                        if cookie: cookie_used = True
                        loop = asyncio.get_running_loop()
                        info = await loop.run_in_executor(self.thread_pool, self._sync_api_extract, url, cookie)
                        if info and info.get("formats"): return info, method_used
                        last_err = "No video formats found in Fast API"

                    # ATTEMPT 3: Ultimate Emergency No-Cookie Fallback (Android/TV)
                    elif attempt == 3:
                        method_used = "CLI Fallback No-Cookies (Android Client)"
                        cookie_used = False 
                        cmd = [YT_DLP_BIN, "--dump-json", url, "--no-warnings", "--remote-components", "ejs:github"]
                        if self.impersonate_supported: cmd.extend(["--impersonate", IMPERSONATE_TARGET])
                        cmd.extend(["--extractor-args", "youtube:player_client=android,tv"])
                        out, err = await self._exec_cmd(*cmd, timeout=25)
                        if out:
                            info = _loads_bytes(out)
                            if info.get("formats"): return info, method_used
                        last_err = err.decode("utf-8", "ignore") if err else "Empty response"

                except Exception as e:
                    last_err = str(e)

                logger.warning(f"⚠️ {method_used} Failed: {last_err.strip()[:200]}")

                if cookie_used and cookie and ("Sign in to confirm" in last_err or "Sign in to confirm you're not a bot" in last_err):
                    await cookie_vault.report_failure(cookie)

                await asyncio.sleep(min(2 ** attempt + random.random(), 5))

        raise Exception(f"YouTube refused connection after attempts. Last error: {last_err}")

titan_core = TitanExtractor()

# ============== FastAPI app & auth ==============
app = FastAPI(title="TitanOS Enterprise Edge API", version="6.6.0-Speed", default_response_class=ORJSONResponse, docs_url=None, redoc_url=None)
app.add_middleware(GZipMiddleware, minimum_size=500)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

api_key_header = APIKeyHeader(name="X-Titan-Key", auto_error=False)

def verify_auth_token(key: Optional[str]) -> str:
    if not key or not _secrets.compare_digest(str(key), str(API_KEY)):
        raise HTTPException(status_code=403, detail="Unauthorized Access")
    return key

async def verify_auth(key: str = Security(api_key_header)):
    return verify_auth_token(key)

@app.on_event("startup")
async def startup_event():
    app.state._bg_task = asyncio.create_task(_background_tasks())

async def _background_tasks():
    try:
        while True:
            await asyncio.sleep(300)
            await cache_engine.cleanup()
            cookie_vault.refresh_pool_sync()
    except asyncio.CancelledError:
        pass

@app.on_event("shutdown")
async def shutdown_event():
    task = getattr(app.state, "_bg_task", None)
    if task:
        task.cancel()

@app.get("/", tags=["System"])
async def index():
    return ORJSONResponse({"system": "TitanOS Edge", "status": "Operational", "mode": "High-Speed Direct Engine"})

@app.get("/api/v1/extract", response_model=MediaResponse, tags=["Media"])
async def extract_media(url: str = Query(...), audio_only: bool = Query(True), force_refresh: bool = Query(False), auth: str = Depends(verify_auth)):
    t0 = time.perf_counter()
    cache_key = f"{url}_audio:{audio_only}"

    try:
        if not force_refresh:
            cached = await cache_engine.get(cache_key)
            if cached:
                cached['cached'] = True
                cached['process_time_ms'] = round((time.perf_counter() - t0) * 1000, 3)
                return ORJSONResponse(cached)

        info, method = await titan_core.extract_smart(url, audio_only)

        direct_url = info.get("url") or ""
        fallback_list: List[FormatModel] = []
        for f in info.get("formats", []):
            proto = str(f.get("protocol", "")).lower()
            if proto.startswith(("http", "https", "m3u8")) or f.get("url"):
                fmt = FormatModel(
                    format_id=str(f.get("format_id", "0")),
                    ext=f.get("ext", "unknown"),
                    resolution=f.get("format_note") or f.get("resolution") or "audio",
                    vcodec=f.get("vcodec") or "none",
                    acodec=f.get("acodec") or "none",
                    url=f.get("url") or ""
                )
                fallback_list.append(fmt)
                if not direct_url:
                    if audio_only and fmt.vcodec == "none" and fmt.acodec != "none": direct_url = fmt.url
                    elif not audio_only and fmt.vcodec != "none" and fmt.acodec != "none": direct_url = fmt.url

        if not direct_url and fallback_list:
            fallback_list.sort(key=lambda x: ("m3u8" in x.url, x.ext, x.format_id))
            direct_url = fallback_list[-1].url if fallback_list else ""

        if not direct_url: raise ValueError("No playable stream found in YouTube response.")

        thumbs = [ThumbnailModel(url=t.get("url"), width=t.get("width", 0), height=t.get("height", 0)) for t in info.get("thumbnails", [])]
        is_live = bool(info.get("is_live") or info.get("was_live"))
        dur = 0 if is_live or str(info.get("duration")).lower() in ["live", "none"] else int(info.get("duration") or 0)

        response_data = {
            "success": True,
            "process_time_ms": round((time.perf_counter() - t0) * 1000, 3),
            "cached": False,
            "extraction_method": method,
            "video_id": info.get("id", ""),
            "title": info.get("title", "Unknown"),
            "duration": dur,
            "is_live": is_live,
            "thumbnails": [model_to_primitive(t) for t in thumbs[-1:]],
            "direct_stream_url": direct_url,
            "fallback_streams": [model_to_primitive(f) for f in fallback_list[-3:]]
        }

        await cache_engine.set(cache_key, response_data)
        return ORJSONResponse(response_data)

    except Exception as e:
        logger.error(f"❌ Extraction Fatal Error for {url}: {str(e)}")
        return ORJSONResponse(status_code=500, content=ErrorResponse(success=False, error_code=500, message=str(e), process_time_ms=round((time.perf_counter() - t0) * 1000, 3)).dict())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8080")), loop="uvloop", http="httptools", workers=int(os.getenv("UVICORN_WORKERS", "4")))

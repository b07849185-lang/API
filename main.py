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
from fastapi.responses import ORJSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel

try:
    import orjson as _orjson  
    def _loads_bytes(b: bytes) -> dict:
        try:
            return _orjson.loads(b)
        except:
            return {}
except Exception:
    import json as _json  
    def _loads_bytes(b: bytes) -> dict:
        try:
            return _json.loads(b.decode("utf-8", "ignore") if isinstance(b, bytes) else b)
        except:
            return {}

try:
    import yt_dlp
except ImportError:
    pass

os.environ.setdefault("TZ", "UTC")
API_KEY = os.getenv("TITAN_SECRET_KEY", "Titan_2026_Ultra_Fast")
COOKIE_DIR = os.getenv("TITAN_COOKIE_DIR", "cookies")
MAX_CONCURRENT_EXTRACTS = int(os.getenv("MAX_THREADS", "8"))  
CACHE_TTL = int(os.getenv("CACHE_TTL", "14400"))
COOKIE_BAN_TIME = int(os.getenv("COOKIE_BAN_TIME", "3600"))
IMPERSONATE_TARGET = os.getenv("YT_IMPERSONATE_TARGET", "chrome110")
YT_DLP_BIN = os.getenv("YT_DLP_BIN", "yt-dlp")

try:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    logger = logging.getLogger("TitanAPI")
except Exception:
    logger = None

class FormatModel(BaseModel):
    format_id: str
    ext: str
    resolution: Optional[str] = "audio only"
    vcodec: str
    acodec: str
    url: str
    protocol: str
    tbr: float

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
    try:
        if hasattr(m, "model_dump"):
            return m.model_dump()
        return m.dict()
    except Exception:
        return {}

class EnterpriseCookieManager:
    def __init__(self, directory: str = COOKIE_DIR, ban_time: int = COOKIE_BAN_TIME, reuse_delay: int = 5):
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
                if not item: return None
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
        self.thread_pool = ThreadPoolExecutor(max_workers=max(4, min(MAX_CONCURRENT_EXTRACTS, 64)))
        self.sema = asyncio.Semaphore(MAX_CONCURRENT_EXTRACTS)
        self.impersonate_supported = False
        try:
            out = self._run_cmd_sync([YT_DLP_BIN, "--list-impersonate-targets"])
            if out and ("chrome" in out or IMPERSONATE_TARGET in out):
                self.impersonate_supported = True
        except Exception:
            pass

    def _run_cmd_sync(self, cmd: List[str], timeout: int = 15) -> str:
        try:
            import subprocess
            p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
            return p.stdout.decode("utf-8", "ignore")
        except Exception:
            return ""

    async def _exec_cmd(self, *args: str, timeout: int = 40) -> Tuple[bytes, bytes]:
        try:
            proc = await asyncio.create_subprocess_exec(
                *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            try:
                out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                return out or b"", err or b""
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                except Exception:
                    pass
                return b"", b"timeout"
        except Exception as e:
            return b"", str(e).encode()

    def _sync_api_extract(self, url: str, cookie: Optional[str]) -> dict:
        try:
            opts = {
                "quiet": True, "no_warnings": True, "skip_download": True, "extract_flat": False, "noplaylist": True,
                "extractor_args": {"youtube": {"player_client": ["web"]}},
            }
            if cookie: 
                opts["cookiefile"] = cookie
            if self.impersonate_supported: 
                opts["impersonate"] = IMPERSONATE_TARGET
            with yt_dlp.YoutubeDL(opts) as ydl:
                return ydl.extract_info(url, download=False)
        except Exception:
            return {}

    async def extract_smart(self, url: str, audio_only: bool) -> Tuple[Dict[str, Any], str]:
        last_err = "unknown"
        async with self.sema:
            for attempt in range(1, 7):  
                cookie = await cookie_vault.get_cookie()
                method_used = ""
                cookie_used = False 
                try:
                    if attempt == 1:
                        method_used = "CLI Standard (Web Client + Remote EJS)"
                        cmd = [YT_DLP_BIN, "--dump-json", url, "--no-warnings", "--remote-components", "ejs:github"]
                        if cookie: 
                            cmd.extend(["--cookies", cookie])
                            cookie_used = True
                        if self.impersonate_supported: 
                            cmd.extend(["--impersonate", IMPERSONATE_TARGET])
                        cmd.extend(["--extractor-args", "youtube:player_client=web"])
                        out, err = await self._exec_cmd(*cmd, timeout=20)
                        if out:
                            info = _loads_bytes(out)
                            if info and info.get("formats"): 
                                return info, method_used
                        last_err = err.decode("utf-8", "ignore") if err else "Empty response"

                    elif attempt == 2:
                        method_used = "CLI iOS Client (Remote EJS)"
                        cmd = [YT_DLP_BIN, "--dump-json", url, "--no-warnings", "--remote-components", "ejs:github"]
                        if cookie: 
                            cmd.extend(["--cookies", cookie])
                            cookie_used = True
                        if self.impersonate_supported: 
                            cmd.extend(["--impersonate", IMPERSONATE_TARGET])
                        cmd.extend(["--extractor-args", "youtube:player_client=ios"])
                        out, err = await self._exec_cmd(*cmd, timeout=25)
                        if out:
                            info = _loads_bytes(out)
                            if info and info.get("formats"): 
                                return info, method_used
                        last_err = err.decode("utf-8", "ignore") if err else "Empty response"

                    elif attempt == 3:
                        method_used = "Fast API (Web Client + Cookie)"
                        if cookie: 
                            cookie_used = True
                        try:
                            loop = asyncio.get_running_loop()
                            info = await loop.run_in_executor(self.thread_pool, self._sync_api_extract, url, cookie)
                            if info and info.get("formats"): 
                                return info, method_used
                        except Exception as inner_e:
                            last_err = str(inner_e)

                    elif attempt == 4:
                        method_used = "CLI Fallback No-Cookies (Android Client)"
                        cookie_used = False 
                        cmd = [YT_DLP_BIN, "--dump-json", url, "--no-warnings", "--remote-components", "ejs:github"]
                        if self.impersonate_supported: 
                            cmd.extend(["--impersonate", IMPERSONATE_TARGET])
                        cmd.extend(["--extractor-args", "youtube:player_client=android"])
                        out, err = await self._exec_cmd(*cmd, timeout=25)
                        if out:
                            info = _loads_bytes(out)
                            if info and info.get("formats"): 
                                return info, method_used
                        last_err = err.decode("utf-8", "ignore") if err else "Empty response"
                        
                    elif attempt == 5:
                        method_used = "CLI Fallback No-Cookies (TV Client)"
                        cookie_used = False 
                        cmd = [YT_DLP_BIN, "--dump-json", url, "--no-warnings", "--remote-components", "ejs:github"]
                        cmd.extend(["--extractor-args", "youtube:player_client=tv"])
                        out, err = await self._exec_cmd(*cmd, timeout=30)
                        if out:
                            info = _loads_bytes(out)
                            if info and info.get("formats"): 
                                return info, method_used
                        last_err = err.decode("utf-8", "ignore") if err else "Empty response"
                        
                    elif attempt == 6:
                        method_used = "Pure Native Fallback"
                        cookie_used = False 
                        cmd = [YT_DLP_BIN, "--dump-json", url, "--no-warnings"]
                        out, err = await self._exec_cmd(*cmd, timeout=35)
                        if out:
                            info = _loads_bytes(out)
                            if info and info.get("formats"): 
                                return info, method_used
                        last_err = err.decode("utf-8", "ignore") if err else "Empty response"

                except Exception as e:
                    last_err = str(e)

                try:
                    if cookie_used and cookie and ("Sign in" in last_err or "bot" in last_err.lower()):
                        await cookie_vault.report_failure(cookie)
                except Exception:
                    pass

                try:
                    await asyncio.sleep(min(2 ** attempt + random.random(), 6))
                except Exception:
                    pass

        raise Exception(f"Extraction failed after all attempts. Last error: {last_err}")

try:
    titan_core = TitanExtractor()
except Exception:
    titan_core = None

app = FastAPI(title="TitanOS Enterprise", version="7.0.0", default_response_class=ORJSONResponse, docs_url=None, redoc_url=None)

try:
    app.add_middleware(GZipMiddleware, minimum_size=500)
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
                await asyncio.sleep(300)
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
        return ORJSONResponse({"system": "TitanOS Core", "status": "Active", "engine": "Ultimate CFFI/JS"})
    except Exception:
        return ORJSONResponse({"status": "Active"})

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

        info, method = await titan_core.extract_smart(url, audio_only)

        direct_url = ""
        try:
            direct_url = info.get("url", "")
        except Exception:
            pass
            
        fallback_list: List[FormatModel] = []
        try:
            formats_data = info.get("formats", [])
            if not isinstance(formats_data, list):
                formats_data = []
                
            for f in formats_data:
                try:
                    if not isinstance(f, dict): continue
                    proto = str(f.get("protocol", "")).lower()
                    f_url = str(f.get("url", ""))
                    if proto.startswith(("http", "https", "m3u8")) or f_url:
                        fmt = FormatModel(
                            format_id=str(f.get("format_id", "0")),
                            ext=str(f.get("ext", "unknown")),
                            resolution=str(f.get("format_note", f.get("resolution", "audio"))),
                            vcodec=str(f.get("vcodec", "none")),
                            acodec=str(f.get("acodec", "none")),
                            url=f_url,
                            protocol=proto,
                            tbr=float(f.get("tbr", 0.0) or 0.0)
                        )
                        fallback_list.append(fmt)
                except Exception:
                    continue
        except Exception:
            pass

        try:
            if fallback_list:
                if audio_only:
                    audio_formats = [f for f in fallback_list if f.vcodec == "none" and f.acodec != "none"]
                    if audio_formats:
                        audio_formats.sort(key=lambda x: (x.tbr, "m4a" in x.ext))
                        direct_url = audio_formats[-1].url
                else:
                    video_formats = [f for f in fallback_list if f.vcodec != "none" and f.acodec != "none"]
                    if video_formats:
                        video_formats.sort(key=lambda x: ("m3u8" in x.protocol, x.tbr))
                        direct_url = video_formats[-1].url
                        
                if not direct_url:
                    fallback_list.sort(key=lambda x: ("m3u8" in x.protocol, x.tbr))
                    direct_url = fallback_list[-1].url
        except Exception:
            if fallback_list and not direct_url:
                direct_url = fallback_list[-1].url

        if not direct_url: 
            raise ValueError("Extraction yielded no valid stream URL")

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
                "thumbnails": [model_to_primitive(t) for t in thumbs[-1:]] if thumbs else [],
                "direct_stream_url": str(direct_url),
                "fallback_streams": [model_to_primitive(f) for f in fallback_list[-5:]] if fallback_list else []
            }
        except Exception as e:
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
            return ORJSONResponse(status_code=500, content=ErrorResponse(success=False, error_code=500, message=str(e), process_time_ms=p_time).dict())
        except Exception:
            return ORJSONResponse(status_code=500, content={"success": False, "error_code": 500, "message": "Critical Failure", "process_time_ms": 0.0})

if __name__ == "__main__":
    try:
        import uvicorn
        uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8080")), loop="uvloop", http="httptools", workers=int(os.getenv("UVICORN_WORKERS", "4")))
    except Exception:
        pass

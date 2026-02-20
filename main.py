"""
TitanOS - hardened extractor service (2026 Gold Release)
Features:
- API key auth (constant-time compare)
- Cookie pool with temporary bans & reuse-delay
- Proxy pool with rotation, per-proxy cooldown & bans
- Multi-strategy yt-dlp extraction (API, CLI, remote, no-cookies)
- Backoff, jitter, rate-limiting, reduced default concurrency
- Smart Bypassing: iOS/Android clients, Webpage skip
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

# JSON loader (orjson if present)
try:
    import orjson as _orjson  # type: ignore
    def _loads_bytes(b: bytes) -> dict:
        return _orjson.loads(b)
except Exception:
    import json as _json  # type: ignore
    def _loads_bytes(b: bytes) -> dict:
        return _json.loads(b.decode("utf-8", "ignore") if isinstance(b, bytes) else b)

import yt_dlp  # requires yt-dlp installed in image

# ============== configuration ==============
os.environ.setdefault("TZ", "UTC")
API_KEY = os.getenv("TITAN_SECRET_KEY", "Titan_2026_Ultra_Fast")
COOKIE_DIR = os.getenv("TITAN_COOKIE_DIR", "cookies")
PROXY_FILE = os.getenv("TITAN_PROXY_FILE", "proxies.txt")  
MAX_CONCURRENT_EXTRACTS = int(os.getenv("MAX_THREADS", "4"))  
CACHE_TTL = int(os.getenv("CACHE_TTL", "14400"))
PROXY_COOLDOWN = int(os.getenv("PROXY_COOLDOWN", "10"))  
PROXY_BAN_TIME = int(os.getenv("PROXY_BAN_TIME", "3600"))
COOKIE_BAN_TIME = int(os.getenv("COOKIE_BAN_TIME", "3600"))
# تحديث هدف المتصفح ليتوافق مع أحدث إصدارات 2026
IMPERSONATE_TARGET = os.getenv("YT_IMPERSONATE_TARGET", "chrome110")
YT_DLP_BIN = os.getenv("YT_DLP_BIN", "yt-dlp")  

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("TitanAPI")

# ============== models ==============
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

# ============== Cookie manager ==============
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
        logger.info(f"Cookie Pool: Active: {len(self.pool)}, Banned: {len(self.banned)}")

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
            logger.warning(f"🚫 Cookie Banned (Temp): {cookie_path}")

cookie_vault = EnterpriseCookieManager()

# ============== Proxy manager ==============
class ProxyManager:
    def __init__(self, proxy_file: str = PROXY_FILE, cooldown: int = PROXY_COOLDOWN, ban_time: int = PROXY_BAN_TIME):
        self.file = proxy_file
        self.cooldown = cooldown
        self.ban_time = ban_time
        self.pool: List[str] = []
        self.last_used: Dict[str, float] = {}
        self.banned: Dict[str, float] = {}
        self._lock = asyncio.Lock()
        self.refresh_sync()

    def refresh_sync(self):
        if os.path.exists(self.file):
            with open(self.file, "r", encoding="utf-8") as fh:
                lines = [l.strip() for l in fh if l.strip() and not l.strip().startswith("#")]
            now = time.time()
            for p in list(self.banned.keys()):
                if now - self.banned[p] > self.ban_time:
                    del self.banned[p]
            self.pool = [p for p in lines if p not in self.banned]
        else:
            self.pool = []
        logger.info(f"Proxy Pool: Active: {len(self.pool)}, Banned: {len(self.banned)}")

    async def get_proxy(self) -> Optional[str]:
        async with self._lock:
            if not self.pool:
                self.refresh_sync()
            if not self.pool:
                return None
            candidates = [p for p in self.pool if (time.time() - self.last_used.get(p, 0)) > self.cooldown]
            if not candidates:
                candidates = self.pool
            p = random.choice(candidates)
            self.last_used[p] = time.time()
            return p

    async def report_failure(self, proxy: str):
        async with self._lock:
            self.banned[proxy] = time.time()
            if proxy in self.pool:
                self.pool.remove(proxy)
            logger.warning(f"🚫 Proxy Banned (Temp): {proxy}")

proxy_manager = ProxyManager()

# ============== Memory cache ==============
class MemoryCache:
    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            item = self._cache.get(key)
            if not item:
                return None
            if time.time() - item["timestamp"] < CACHE_TTL:
                return item["data"]
            del self._cache[key]
            return None

    async def set(self, key: str, data: Dict[str, Any]):
        async with self._lock:
            self._cache[key] = {"timestamp": time.time(), "data": data}

    async def cleanup(self):
        async with self._lock:
            now = time.time()
            expired = [k for k, v in self._cache.items() if now - v["timestamp"] >= CACHE_TTL]
            for k in expired:
                del self._cache[k]

cache_engine = MemoryCache()

# ============== Titan extractor ==============
class TitanExtractor:
    def __init__(self):
        self.thread_pool = ThreadPoolExecutor(max_workers=max(2, min(MAX_CONCURRENT_EXTRACTS, 32)))
        self.sema = asyncio.Semaphore(MAX_CONCURRENT_EXTRACTS)
        self.impersonate_supported = False
        self.available_impersonate_targets: List[str] = []
        try:
            out = self._run_cmd_sync([YT_DLP_BIN, "--list-impersonate-targets"])
            if out:
                self.available_impersonate_targets = out.splitlines()
                # Use partial match to ensure it finds chrome110, etc.
                if any(IMPERSONATE_TARGET in t for t in self.available_impersonate_targets) or any("chrome" in t for t in self.available_impersonate_targets):
                    self.impersonate_supported = True
        except Exception:
            self.impersonate_supported = False

    def _run_cmd_sync(self, cmd: List[str], timeout: int = 10) -> str:
        import subprocess
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        out = p.stdout.decode("utf-8", "ignore")
        return out

    async def _exec_cmd(self, *args: str, timeout: int = 25, env: Optional[dict] = None) -> Tuple[bytes, bytes]:
        proc = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=env
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            return out or b"", err or b""
        except asyncio.TimeoutError:
            with contextlib.suppress(Exception):
                proc.kill()
            return b"", b"timeout"

    def _sync_api_extract(self, url: str, cookie: Optional[str], proxy: Optional[str]) -> dict:
        opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "extract_flat": False,
            "noplaylist": True,
            # 🚀 التعديل الأهم: إضافة webpage و js لتجاوز رسالة The page needs to be reloaded
            "extractor_args": {"youtube": {"player_client": ["ios", "android", "tv"], "player_skip": ["webpage", "js", "configs"]}},
        }
        if cookie:
            opts["cookiefile"] = cookie
        if self.impersonate_supported:
            opts["impersonate"] = IMPERSONATE_TARGET
        if proxy:
            opts["proxy"] = proxy

        with yt_dlp.YoutubeDL(opts) as ydl:
            return ydl.extract_info(url, download=False)

    async def extract_smart(self, url: str, audio_only: bool) -> Tuple[Dict[str, Any], str]:
        last_err = "unknown"
        async with self.sema:
            for attempt in range(1, 6):
                cookie = await cookie_vault.get_cookie()
                proxy = await proxy_manager.get_proxy()
                method_used = ""
                env = os.environ.copy()
                if proxy:
                    env.update({"http_proxy": proxy, "https_proxy": proxy, "HTTP_PROXY": proxy, "HTTPS_PROXY": proxy})

                try:
                    if attempt == 1:
                        method_used = "Fast API (yt-dlp Python API)"
                        loop = asyncio.get_running_loop()
                        info = await loop.run_in_executor(self.thread_pool, self._sync_api_extract, url, cookie, proxy)
                        if info and info.get("formats"):
                            return info, method_used

                    elif attempt == 2:
                        method_used = "CLI IPv6 (yt-dlp) with proxy"
                        cmd = [YT_DLP_BIN, "--dump-json", url, "--no-warnings", "--force-ipv6"]
                        if cookie: cmd.extend(["--cookies", cookie])
                        if proxy: cmd.extend(["--proxy", proxy])
                        if self.impersonate_supported: cmd.extend(["--impersonate", IMPERSONATE_TARGET])
                        cmd.extend(["--extractor-args", "youtube:player_client=android,ios;player_skip=webpage,configs,js"])
                        out, err = await self._exec_cmd(*cmd, timeout=30, env=env)
                        if out:
                            info = _loads_bytes(out)
                            if info.get("formats"): return info, method_used
                        last_err = (err.decode("utf-8", "ignore") if err else "Empty response")

                    elif attempt == 3:
                        method_used = "CLI Remote (ejs) with proxy"
                        cmd = [YT_DLP_BIN, "--dump-json", url, "--no-warnings", "--remote-components", "ejs:github"]
                        if cookie: cmd.extend(["--cookies", cookie])
                        if proxy: cmd.extend(["--proxy", proxy])
                        if self.impersonate_supported: cmd.extend(["--impersonate", IMPERSONATE_TARGET])
                        cmd.extend(["--extractor-args", "youtube:player_skip=webpage,configs,js"])
                        out, err = await self._exec_cmd(*cmd, timeout=35, env=env)
                        if out:
                            info = _loads_bytes(out)
                            if info.get("formats"): return info, method_used
                        last_err = (err.decode("utf-8", "ignore") if err else "Empty response")

                    elif attempt == 4:
                        method_used = "CLI No-Cookies (Fallback) with proxy"
                        cmd = [YT_DLP_BIN, "--dump-json", url, "--no-warnings", "--extractor-args", "youtube:player_client=tv;player_skip=webpage,configs,js"]
                        if proxy: cmd.extend(["--proxy", proxy])
                        out, err = await self._exec_cmd(*cmd, timeout=30, env=env)
                        if out:
                            info = _loads_bytes(out)
                            if info.get("formats"): return info, method_used
                        last_err = (err.decode("utf-8", "ignore") if err else "Empty response")

                    else:
                        method_used = "Final attempt - exhaustive (no cookies, no impersonate)"
                        cmd = [YT_DLP_BIN, "--dump-json", url, "--no-warnings", "--extractor-args", "youtube:player_skip=webpage,configs,js"]
                        out, err = await self._exec_cmd(*cmd, timeout=35, env=env)
                        if out:
                            info = _loads_bytes(out)
                            if info.get("formats"): return info, method_used
                        last_err = (err.decode("utf-8", "ignore") if err else "Empty response")

                except Exception as e:
                    last_err = str(e)

                logger.warning(f"⚠️ {method_used} Failed: {last_err.strip()[:300]}")

                # 🚀 التعديل الأهم: إزالة "The page needs to be reloaded" من أسباب الحظر لحماية الكوكيز
                if cookie and ("Sign in to confirm" in last_err or "Sign in to confirm you're not a bot" in last_err):
                    await cookie_vault.report_failure(cookie)

                if proxy and ("403" in last_err or "timed out" in last_err.lower() or "connection refused" in last_err.lower()):
                    await proxy_manager.report_failure(proxy)

                await asyncio.sleep(min(2 ** attempt + random.random(), 12))

        raise Exception(f"YouTube refused connection after attempts. Last error: {last_err}")

titan_core = TitanExtractor()

# ============== FastAPI app & auth ==============
app = FastAPI(title="TitanOS Enterprise Edge API", version="6.1.0-Gold", default_response_class=ORJSONResponse, docs_url=None, redoc_url=None)
app.add_middleware(GZipMiddleware, minimum_size=500)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

api_key_header = APIKeyHeader(name="X-Titan-Key", auto_error=False)

def verify_auth_token(key: Optional[str]) -> str:
    if not key:
        raise HTTPException(status_code=403, detail="Unauthorized Access")
    if not _secrets.compare_digest(str(key), str(API_KEY)):
        raise HTTPException(status_code=403, detail="Unauthorized Access")
    return key

async def verify_auth(key: str = Security(api_key_header)):
    return verify_auth_token(key)

# background tasks
@app.on_event("startup")
async def startup_event():
    app.state._bg_task = asyncio.create_task(_background_tasks())

async def _background_tasks():
    try:
        while True:
            await asyncio.sleep(300)
            await cache_engine.cleanup()
            cookie_vault.refresh_pool_sync()
            proxy_manager.refresh_sync()
    except asyncio.CancelledError:
        raise

@app.on_event("shutdown")
async def shutdown_event():
    task = getattr(app.state, "_bg_task", None)
    if task:
        task.cancel()
        with contextlib.suppress(Exception):
            asyncio.get_event_loop().run_until_complete(task)

@app.get("/", tags=["System"])
async def index():
    return ORJSONResponse({"system": "TitanOS Edge", "status": "Operational", "cores_optimized": True})

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
                    if audio_only and fmt.vcodec == "none" and fmt.acodec != "none":
                        direct_url = fmt.url
                    elif not audio_only and fmt.vcodec != "none" and fmt.acodec != "none":
                        direct_url = fmt.url

        if not direct_url and fallback_list:
            fallback_list.sort(key=lambda x: ("m3u8" in x.url, x.ext, x.format_id))
            direct_url = fallback_list[-1].url if fallback_list else ""

        if not direct_url:
            raise ValueError("No playable stream found in YouTube response.")

        thumbs = [ThumbnailModel(url=t.get("url"), width=t.get("width", 0), height=t.get("height", 0)) for t in info.get("thumbnails", [])]
        is_live = bool(info.get("is_live") or info.get("was_live"))
        dur_str = info.get("duration")
        dur = 0 if is_live or str(dur_str).lower() in ["live", "none"] else int(dur_str or 0)

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
        await asyncio.sleep(random.uniform(0.5, 1.5))
        return ORJSONResponse(response_data)

    except Exception as e:
        logger.error(f"❌ Extraction Fatal Error for {url}: {str(e)}")
        return ORJSONResponse(status_code=500, content=ErrorResponse(success=False, error_code=500, message=str(e), process_time_ms=round((time.perf_counter() - t0) * 1000, 3)).dict())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8080")), loop="uvloop", http="httptools", workers=int(os.getenv("UVICORN_WORKERS", "1")))

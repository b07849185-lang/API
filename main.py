import os
import time
import glob
import random
import asyncio
import logging
import contextlib
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Dict, Any, Tuple

from fastapi import FastAPI, HTTPException, Request, Depends, Query, Security
from fastapi.responses import ORJSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel

try:
    import orjson
except ImportError:
    import json as orjson

import yt_dlp

# ==========================================
# إعدادات النظام الأساسية
# ==========================================
os.environ["TZ"] = "UTC"
API_KEY = os.getenv("TITAN_SECRET_KEY", "Titan_2026_Ultra_Fast")
MAX_CONCURRENT_EXTRACTS = int(os.getenv("MAX_THREADS", "30")) # حماية المعالج من الاختناق
CACHE_TTL = 14400  # 4 ساعات كاش لتقليل الضغط

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("TitanAPI")

# ==========================================
# Pydantic Models (لضمان صحة وسرعة البيانات)
# ==========================================
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

# ==========================================
# إدارة الكوكيز (Enterprise Grade)
# ==========================================
class EnterpriseCookieManager:
    def __init__(self, directory: str = "cookies"):
        self.directory = directory
        self.pool: List[str] = []
        self.banned: Dict[str, float] = {}
        self.ban_time = 3600  # تجميد الكوكيز المحظور لمدة ساعة
        self._lock = asyncio.Lock()
        self._ensure_dir()
        self.refresh_pool_sync()

    def _ensure_dir(self):
        if not os.path.exists(self.directory):
            os.makedirs(self.directory, exist_ok=True)

    def refresh_pool_sync(self):
        files = glob.glob(os.path.join(self.directory, "*.txt"))
        now = time.time()
        for cookie in list(self.banned.keys()):
            if now - self.banned[cookie] > self.ban_time:
                del self.banned[cookie]
        self.pool = [f for f in files if f not in self.banned and os.path.getsize(f) > 0]
        logger.info(f"Cookie Pool: Active: {len(self.pool)}, Banned: {len(self.banned)}")

    async def get_cookie(self) -> Optional[str]:
        async with self._lock:
            if not self.pool:
                self.refresh_pool_sync()
            if not self.pool:
                return None
            return random.choice(self.pool)

    async def report_failure(self, cookie_path: str):
        async with self._lock:
            if cookie_path in self.pool:
                self.banned[cookie_path] = time.time()
                self.pool.remove(cookie_path)
                logger.warning(f"🚫 Cookie Banned (Temp): {cookie_path}")

cookie_vault = EnterpriseCookieManager()

# ==========================================
# محرك الكاش (لتقليل وقت الاستجابة لـ 0.01ms)
# ==========================================
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

# ==========================================
# محرك الاستخراج (الوحش الذي لا يموت)
# ==========================================
class TitanExtractor:
    def __init__(self):
        self.thread_pool = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_EXTRACTS)
        self.sema = asyncio.Semaphore(MAX_CONCURRENT_EXTRACTS)
        try:
            import curl_cffi
            self.impersonate = True
        except ImportError:
            self.impersonate = False

    async def _exec_cmd(self, *args: str, timeout: int = 15) -> Tuple[bytes, bytes]:
        """المسار البطيء والأكثر قوة (Subprocess)"""
        proc = await asyncio.create_subprocess_exec(
            *args, 
            stdout=asyncio.subprocess.PIPE, 
            stderr=asyncio.subprocess.PIPE
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            return out, err
        except asyncio.TimeoutError:
            with contextlib.suppress(Exception):
                proc.kill()
            return b"", b"timeout"

    def _sync_api_extract(self, url: str, cookie: str) -> dict:
        """المسار السريع جداً (Python API)"""
        opts = {
            'quiet': True, 'no_warnings': True, 'skip_download': True,
            'extract_flat': False, 'noplaylist': True,
            'extractor_args': {'youtube': {'player_client': ['tv', 'android'], 'player_skip': ['js', 'configs']}}
        }
        if cookie: opts['cookiefile'] = cookie
        if self.impersonate: opts['impersonate'] = 'chrome'

        with yt_dlp.YoutubeDL(opts) as ydl:
            return ydl.extract_info(url, download=False)

    async def extract_smart(self, url: str, audio_only: bool) -> Tuple[Dict[str, Any], str]:
        last_err = ""
        
        async with self.sema:
            # 🚀 استراتيجية الهجوم متعدد الطبقات 🚀
            for attempt in range(1, 5):
                cookie = await cookie_vault.get_cookie()
                method_used = ""

                try:
                    # التكتيك 1: المسار السريع (API) - يعمل في 600ms
                    if attempt == 1:
                        method_used = "Fast API (TV Client)"
                        loop = asyncio.get_running_loop()
                        info = await loop.run_in_executor(self.thread_pool, self._sync_api_extract, url, cookie)
                        if info and info.get("formats"): return info, method_used

                    # التكتيك 2: مسار الطوارئ 1 (IPv6 CLI + Android) - يتخطى حظر الداتا سنتر
                    elif attempt == 2:
                        method_used = "CLI IPv6 (Android)"
                        cmd = ["yt-dlp", "--dump-json", url, "--no-warnings", "--force-ipv6"]
                        if cookie: cmd.extend(["--cookies", cookie])
                        if self.impersonate: cmd.extend(["--impersonate", "chrome"])
                        cmd.extend(["--extractor-args", "youtube:player_client=android,web;player_skip=configs"])
                        
                        out, err = await self._exec_cmd(*cmd)
                        if out:
                            info = orjson.loads(out)
                            if info.get("formats"): return info, method_used
                        last_err = err.decode('utf-8', 'ignore') if err else "Empty response"

                    # التكتيك 3: مسار الطوارئ 2 (Remote Components) - يكسر حماية Reload
                    elif attempt == 3:
                        method_used = "CLI Remote (EJS)"
                        cmd = ["yt-dlp", "--dump-json", url, "--no-warnings", "--remote-components", "ejs:github"]
                        if cookie: cmd.extend(["--cookies", cookie])
                        if self.impersonate: cmd.extend(["--impersonate", "chrome"])
                        
                        out, err = await self._exec_cmd(*cmd, timeout=20)
                        if out:
                            info = orjson.loads(out)
                            if info.get("formats"): return info, method_used
                        last_err = err.decode('utf-8', 'ignore') if err else "Empty response"

                    # التكتيك 4: الهجوم الأخير بدون كوكيز (أحياناً الكوكيز هي سبب الحظر)
                    elif attempt == 4:
                        method_used = "CLI No-Cookies (Fallback)"
                        cmd = ["yt-dlp", "--dump-json", url, "--no-warnings", "--extractor-args", "youtube:player_client=tv"]
                        out, err = await self._exec_cmd(*cmd)
                        if out:
                            info = orjson.loads(out)
                            if info.get("formats"): return info, method_used
                        last_err = err.decode('utf-8', 'ignore') if err else "Empty response"

                except Exception as e:
                    last_err = str(e)

                # إذا فشلنا، احظر الكوكيز وجرب غيره فوراً
                logger.warning(f"⚠️ {method_used} Failed: {last_err.strip()[:100]}...")
                if cookie and attempt < 4:
                    await cookie_vault.report_failure(cookie)
                await asyncio.sleep(0.5)

        raise Exception(f"يوتيوب يرفض الاتصال نهائياً. السبب الأخير: {last_err}")

titan_core = TitanExtractor()

# ==========================================
# إعداد خادم FastAPI
# ==========================================
app = FastAPI(
    title="TitanOS Enterprise Edge API",
    version="6.0.0-Invincible",
    default_response_class=ORJSONResponse,
    docs_url=None, redoc_url=None
)

app.add_middleware(GZipMiddleware, minimum_size=500)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

api_key_header = APIKeyHeader(name="X-Titan-Key", auto_error=False)

async def verify_auth(key: str = Security(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized Access")
    return key

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(background_tasks())

async def background_tasks():
    while True:
        await asyncio.sleep(300) # تنظيف كل 5 دقائق
        await cache_engine.cleanup()
        cookie_vault.refresh_pool_sync()

@app.get("/", tags=["System"])
async def index():
    return ORJSONResponse({"system": "TitanOS Edge", "status": "Operational", "cores_optimized": True})

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

        info, method = await titan_core.extract_smart(url, audio_only)

        direct_url = info.get("url")
        fallback_list = []
        
        for f in info.get("formats", []):
            protocol = str(f.get("protocol", "")).lower()
            if protocol.startswith(("http", "https", "m3u8")):
                fmt_obj = FormatModel(
                    format_id=str(f.get("format_id", "0")),
                    ext=f.get("ext", "unknown"),
                    resolution=f.get("format_note") or f.get("resolution") or "audio",
                    vcodec=f.get("vcodec", "none"),
                    acodec=f.get("acodec", "none"),
                    url=f.get("url", "")
                )
                fallback_list.append(fmt_obj)
                
                # خوارزمية اختيار أفضل رابط للموقع
                if not direct_url:
                    if audio_only and fmt_obj.vcodec == "none" and fmt_obj.acodec != "none":
                        direct_url = fmt_obj.url
                    elif not audio_only and fmt_obj.vcodec != "none" and fmt_obj.acodec != "none":
                        direct_url = fmt_obj.url

        if not direct_url and fallback_list:
            # ترتيب الروابط وتفضيل m3u8 للبث الحي
            fallback_list.sort(key=lambda x: 1 if "m3u8" in x.url else 0)
            direct_url = fallback_list[-1].url 

        if not direct_url:
            raise ValueError("No playable stream found in YouTube response.")

        thumbs = [ThumbnailModel(url=t.get("url"), width=t.get("width", 0), height=t.get("height", 0)) for t in info.get("thumbnails", [])]

        is_live = info.get("is_live") or info.get("was_live") or False
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
            "thumbnails": [t.model_dump() for t in thumbs[-1:]], 
            "direct_stream_url": direct_url,
            "fallback_streams": [f.model_dump() for f in fallback_list[-3:]] 
        }

        await cache_engine.set(cache_key, response_data)
        return ORJSONResponse(response_data)

    except Exception as e:
        logger.error(f"❌ Extraction Fatal Error for {url}: {str(e)}")
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
    # إعدادات جاهزة لتحمل آلاف الطلبات
    uvicorn.run("main:app", host="0.0.0.0", port=8080, loop="uvloop", http="httptools", workers=4)

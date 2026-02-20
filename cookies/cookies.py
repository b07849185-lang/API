import os
import aiohttp
import aiofiles
import logging
import config

# استخدام اللوجر العادي لمنع تداخل الاستدعاءات (Circular Import)
logger = logging.getLogger(__name__)

async def save_cookies():
    cookie_link = getattr(config, "COOKIE_URL", None) or os.getenv("COOKIE_URL")
    
    if not cookie_link:
        logger.warning("⚠️ No COOKIE_URL found. Skipping cookie download.")
        return

    logger.info("🍪 Found COOKIE_URL, downloading...")

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(cookie_link, timeout=10) as response:
                if response.status != 200:
                    logger.error(f"❌ Failed to fetch cookies. HTTP Status: {response.status}")
                    return
                content = await response.text()
        except Exception as e:
            logger.error(f"❌ Cookie Connection Error: {e}")
            return

        if content:
            file_path = "cookies.txt"
            try:
                async with aiofiles.open(file_path, "w") as file:
                    await file.write(content)
                
                if os.path.getsize(file_path) > 0:
                    logger.info(f"✅ Cookies saved successfully to {file_path}.")
                else:
                    logger.error("⚠️ Downloaded cookie file is empty!")
            except Exception as e:
                logger.error(f"❌ Failed to write cookie file: {e}")
        else:
            logger.error("⚠️ Cookie content is empty/null.")

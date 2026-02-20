FROM python:3.12-slim

# إعدادات البيئة لتحسين الأداء ومنع استهلاك الرام في الكاش
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TZ=UTC
# توجيه المكتبات لاستخدام الأنوية الأربعة بالكامل
ENV OMP_NUM_THREADS=4 

WORKDIR /app

# تثبيت الحزم الأساسية + aria2 لتسريع التحميل + build-essential لتسريع بناء المكتبات
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg curl aria2 build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# تحديث pip وتثبيت orjson (أسرع جيسون) والأدوات الأساسية فوراً
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir orjson uvloop httptools && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -U yt-dlp curl_cffi

COPY . .

EXPOSE 8080

# تشغيل السيرفر بأقصى طاقة (4 عمال لـ 4 أنوية)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--loop", "uvloop", "--http", "httptools", "--workers", "4"]
#.

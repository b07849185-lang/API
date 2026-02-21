FROM python:3.12-slim

# إعدادات البيئة لتحسين الأداء ومنع استهلاك الرام في الكاش
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TZ=UTC

# توجيه المكتبات لاستخدام الأنوية بالكامل (بيتم استبدالها من fly.toml)
ENV OMP_NUM_THREADS=4 

# تعريف مسار Deno عشان النظام يشوفه كأمر أساسي
ENV DENO_INSTALL="/root/.deno"
ENV PATH="$DENO_INSTALL/bin:$PATH"

WORKDIR /app

# تثبيت الحزم الأساسية + aria2 + build-essential + git 
# + إضافة Node.js و unzip (مهم لفك ضغط Deno)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg curl aria2 build-essential git nodejs unzip && \
    # تثبيت Deno
    curl -fsSL https://deno.land/install.sh | sh && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# إعداد ملف الكونفيج العام لـ yt-dlp عشان يثبت الـ remote-components إجبارياً
RUN mkdir -p /etc/yt-dlp && \
    echo "--remote-components ejs:github" > /etc/yt-dlp.conf

COPY requirements.txt .

# تحديث pip وتثبيت orjson (أسرع جيسون) والأدوات الأساسية
# + curl_cffi لفك الروابط المستعصية
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir orjson uvloop httptools && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -U yt-dlp curl_cffi

COPY . .

EXPOSE 8080

# 🚀 السحر هنا: استخدام `sh -c` لضمان قراءة المتغير UVICORN_WORKERS من ملف fly.toml (10 أنوية)
# الـ :-4 دي معناها لو المتغير مش موجود لأي سبب، هيشتغل بـ 4 كاحتياطي
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 8080 --loop uvloop --http httptools --workers ${UVICORN_WORKERS:-4}"]

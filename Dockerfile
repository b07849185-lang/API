# ==========================================
# المرحلة الأولى: البناء (Crystal Builder - Debian Based)
# ==========================================
# شلنا كلمة alpine عشان تتوافق مع بيئة البايثون في المرحلة التانية
FROM crystallang/crystal:1.19.1 AS builder

WORKDIR /app
COPY . .

# إنشاء ملف الاعتماديات
RUN echo "name: ultracore" > shard.yml && \
    echo "version: 1.0.0" >> shard.yml && \
    echo "dependencies:" >> shard.yml && \
    echo "  kemal:" >> shard.yml && \
    echo "    github: kemalcr/kemal" >> shard.yml

# تثبيت الاعتماديات
RUN shards install

# بناء السيرفر
RUN crystal build main.cr --release --no-debug -o server

# ==========================================
# المرحلة الثانية: التشغيل النهائي (Python 3.12 Slim - Debian Based)
# ==========================================
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TZ=UTC
ENV OMP_NUM_THREADS=10
ENV DENO_INSTALL="/root/.deno"
ENV PATH="$DENO_INSTALL/bin:$PATH"

WORKDIR /app

# تثبيت الأدوات الأساسية و Node.js
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg curl aria2 build-essential git nodejs unzip && \
    curl -fsSL https://deno.land/install.sh | sh && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# إعداد yt-dlp
RUN mkdir -p /etc/yt-dlp && \
    echo "--remote-components ejs:github" > /etc/yt-dlp.conf

COPY requirements.txt .

# تثبيت مكتبات البايثون (yt-dlp و curl_cffi)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# نقل سيرفر الكريستال المبني من المرحلة الأولى وإعطاؤه صلاحيات التشغيل
COPY --from=builder /app/server .
RUN chmod +x ./server

COPY index.html ./index.html
RUN mkdir -p cookies

EXPOSE 8080

CMD ["./server"]

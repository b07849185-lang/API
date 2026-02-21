# المرحلة الأولى: بناء السيرفر (Crystal Builder)
FROM crystallang/crystal:1.19.1 AS builder

WORKDIR /app
COPY . .

RUN echo "name: ultracore" > shard.yml && \
    echo "version: 1.0.0" >> shard.yml && \
    echo "dependencies:" >> shard.yml && \
    echo "  kemal:" >> shard.yml && \
    echo "    github: kemalcr/kemal" >> shard.yml

RUN shards install
RUN crystal build main.cr --release --no-debug -o server

# المرحلة الثانية: التشغيل (Python 3.12 Slim)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TZ=UTC
ENV OMP_NUM_THREADS=10
ENV DENO_INSTALL="/root/.deno"
ENV PATH="$DENO_INSTALL/bin:$PATH"

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg curl aria2 build-essential git nodejs unzip && \
    curl -fsSL https://deno.land/install.sh | sh && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /etc/yt-dlp && \
    echo "--remote-components ejs:github" > /etc/yt-dlp.conf

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY --from=builder /app/server .
RUN chmod +x ./server

COPY index.html ./index.html

# ⚠️ السطر السحري اللي كان ناقص: نسخ الكوكيز من جهازك للسيرفر ⚠️
COPY cookies/ ./cookies/

EXPOSE 8080
CMD ["./server"]

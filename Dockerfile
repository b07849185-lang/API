# ==========================================
# المرحلة الأولى: بناء سيرفر الـ C++ على أحدث بيئة
# ==========================================
FROM ubuntu:26.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# تثبيت أحدث المترجمات
RUN apt-get update && apt-get install -y \
    g++ cmake make git libssl-dev zlib1g-dev \
    uuid-dev libjsoncpp-dev sqlite3 libsqlite3-dev

# بناء إطار العمل (Drogon)
RUN git clone https://github.com/drogonframework/drogon.git \
    && cd drogon && git submodule update --init \
    && mkdir build && cd build && cmake .. && make -j$(nproc) && make install

WORKDIR /src
COPY API/ ./API/
RUN mkdir -p /src/API/build && cd /src/API/build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release \
    && make -j$(nproc)

# ==========================================
# المرحلة الثانية: التشغيل (Ubuntu 26.04)
# ==========================================
FROM ubuntu:26.04

# إعدادات البيئة بتاعتك لتحسين الأداء
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TZ=UTC
ENV OMP_NUM_THREADS=4 

# تعريف مسار Deno 
ENV DENO_INSTALL="/root/.deno"
ENV PATH="$DENO_INSTALL/bin:$PATH"

WORKDIR /app

# تثبيت الحزم الأساسية بتاعتك + بايثون + متطلبات تشغيل C++
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    ffmpeg curl aria2 build-essential git nodejs unzip \
    libuuid1 zlib1g libjsoncpp-dev \
    && curl -fsSL https://deno.land/install.sh | sh \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# إعداد بيئة بايثون معزولة (إجباري في Ubuntu 26.04 للأمان والسرعة)
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# إعداد ملف الكونفيج العام لـ yt-dlp
RUN mkdir -p /etc/yt-dlp && \
    echo "--remote-components ejs:github" > /etc/yt-dlp.conf

COPY requirements.txt .

# تحديث pip وتثبيت الأدوات اللي إنت طالبها بالظبط
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir orjson uvloop httptools && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -U yt-dlp curl_cffi

COPY . .

# 🚀 السحر الجديد: سحب سيرفر الـ C++ اللي اتبنى في المرحلة الأولى
COPY --from=builder /src/API/build/UltraServer .

EXPOSE 8080

# تشغيل سيرفر الـ C++ الخارق بدل Uvicorn، مع الحفاظ على طريقة الـ sh -c
CMD ["sh", "-c", "./UltraServer"]

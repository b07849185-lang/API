FROM golang:1.24-alpine AS builder
WORKDIR /app
COPY . .
RUN go mod init titanos && \
    go get github.com/gorilla/websocket && \
    CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -o server main.go

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
COPY index.html ./index.html
RUN mkdir -p cookies

EXPOSE 8080

CMD ["./server"]

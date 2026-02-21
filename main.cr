require "kemal"
require "json"
require "file_utils"

API_KEY = ENV.fetch("TITAN_SECRET_KEY", "Titan_2026_Ultra_Fast")
COOKIE_DIR = ENV.fetch("TITAN_COOKIE_DIR", "cookies")
CACHE_TTL = ENV.fetch("CACHE_TTL", "14400").to_f
COOKIE_BAN_TIME = ENV.fetch("COOKIE_BAN_TIME", "3600").to_f
YT_DLP_BIN = ENV.fetch("YT_DLP_BIN", "yt-dlp")
SERVER_START = Time.utc

class StatsStore
  property total_requests : UInt64 = 0
  property successful_requests : UInt64 = 0
  property failed_requests : UInt64 = 0
  property active_ws_connections : Int64 = 0
  @mutex = Mutex.new

  def inc_total
    @mutex.synchronize { @total_requests += 1 }
  end
  def inc_success
    @mutex.synchronize { @successful_requests += 1 }
  end
  def inc_fail
    @mutex.synchronize { @failed_requests += 1 }
  end
  def inc_ws
    @mutex.synchronize { @active_ws_connections += 1 }
  end
  def dec_ws
    @mutex.synchronize { @active_ws_connections -= 1 }
  end
end

STATS = StatsStore.new

class DynamicConfig
  property smart_validation_enabled : Bool = true
  property allow_new_ws_connections : Bool = true
  @mutex = Mutex.new

  def update(smart : Bool?, ws : Bool?)
    @mutex.synchronize do
      @smart_validation_enabled = smart unless smart.nil?
      @allow_new_ws_connections = ws unless ws.nil?
    end
  end
end

CONFIG = DynamicConfig.new

struct StreamInfo
  include JSON::Serializable
  property format_id : String
  property ext : String
  property resolution : String
  property vcodec : String
  property acodec : String
  property url : String
  property quality_score : Int32

  def initialize(@format_id, @ext, @resolution, @vcodec, @acodec, @url, @quality_score)
  end
end

struct SmartFormats
  include JSON::Serializable
  property best_muxed : Array(StreamInfo)
  property audio_only : Array(StreamInfo)
  property video_only : Array(StreamInfo)

  def initialize(@best_muxed, @audio_only, @video_only)
  end
end

struct ThumbnailModel
  include JSON::Serializable
  property url : String
  property width : Int32
  property height : Int32

  def initialize(@url, @width, @height)
  end
end

struct MediaResponse
  include JSON::Serializable
  property success : Bool
  property process_time_ms : Float64
  property cached : Bool
  property extraction_method : String
  property video_id : String
  property title : String
  property duration : Int32
  property is_live : Bool
  property thumbnails : Array(ThumbnailModel)
  property direct_stream_url : String
  property smart_formats : SmartFormats
  property raw_fallback_count : Int32

  def initialize(@success, @process_time_ms, @cached, @extraction_method, @video_id, @title, @duration, @is_live, @thumbnails, @direct_stream_url, @smart_formats, @raw_fallback_count)
  end
end

def get_memory_usage : String
  begin
    lines = File.read("/proc/meminfo").split("\n")
    total = 0
    available = 0
    lines.each do |line|
      if line.starts_with?("MemTotal:")
        total = line.split[1].to_i
      elsif line.starts_with?("MemAvailable:")
        available = line.split[1].to_i
      end
    end
    if total > 0
      used = total - available
      return "#{(used / 1024.0).round(2)} MB / #{(total / 1024.0).round(2)} MB"
    end
  rescue
  end
  "Unknown"
end

class CookieManager
  property pool = Array(String).new
  @banned = Hash(String, Time).new
  @last_used = Hash(String, Time).new
  @mutex = Mutex.new

  def initialize(@dir : String)
    Dir.mkdir_p(@dir) unless Dir.exists?(@dir)
    refresh
  end

  def refresh
    @mutex.synchronize do
      now = Time.utc
      @banned.reject! { |_, time| (now - time).total_seconds > COOKIE_BAN_TIME }
      @pool.clear
      Dir.glob(File.join(@dir, "*.txt")).each do |file|
        if File.info(file).size > 0 && !@banned.has_key?(file)
          @pool << file
        end
      end
    end
  end

  def get_cookie : String?
    @mutex.synchronize do
      return nil if @pool.empty?
      now = Time.utc
      candidates = @pool.select { |c| !@last_used.has_key?(c) || (now - @last_used[c]).total_seconds > 2 }
      candidates = @pool if candidates.empty?
      chosen = candidates.sample
      @last_used[chosen] = now
      chosen
    end
  end

  def report_failure(cookie : String?)
    return unless cookie
    @mutex.synchronize do
      @banned[cookie] = Time.utc
      @pool.delete(cookie)
    end
  end
end

COOKIES = CookieManager.new(COOKIE_DIR)

class MemoryCache
  record Item, timestamp : Time, data : String
  @store = Hash(String, Item).new
  @mutex = Mutex.new

  def get(key : String) : String?
    @mutex.synchronize do
      if item = @store[key]?
        if (Time.utc - item.timestamp).total_seconds < CACHE_TTL
          return item.data
        else
          @store.delete(key)
        end
      end
    end
    nil
  end

  def set(key : String, data : String)
    @mutex.synchronize do
      @store[key] = Item.new(Time.utc, data)
    end
  end

  def clean
    @mutex.synchronize do
      now = Time.utc
      @store.reject! { |_, item| (now - item.timestamp).total_seconds >= CACHE_TTL }
    end
  end

  def clear_all
    @mutex.synchronize { @store.clear }
  end

  def size
    @mutex.synchronize { @store.size }
  end
end

CACHE = MemoryCache.new

def execute_yt_dlp(query : String, cookie : String?, attempt : Int32)
  args = [
    "--dump-json", "--no-warnings", "--skip-download", "--no-playlist",
    "--socket-timeout", "15", "--compat-options", "no-youtube-unavailable-videos",
    "--geo-bypass", "--impersonate", "chrome"
  ]
  
  if cookie && !cookie.empty?
    args << "--cookies"
    args << cookie
  end

  method = ""
  if attempt == 1
    args << "--extractor-args" << "youtube:player_client=web" << "--remote-components" << "ejs:github,ejs:npm"
    method = "Crystal Engine (Web + Chrome)"
  elsif attempt == 2
    args << "--extractor-args" << "youtube:player_client=android,web;player_skip=configs" << "--remote-components" << "ejs:github,ejs:npm"
    method = "Crystal Engine (Android/Web + Chrome)"
  else
    args << "--extractor-args" << "youtube:player_client=web"
    method = "Crystal Engine (Web Fallback)"
  end

  final_query = query
  if !query.starts_with?("http") && !query.starts_with?("ytsearch")
    final_query = "ytsearch1:#{query}"
  end
  args << final_query

  stdout = IO::Memory.new
  stderr = IO::Memory.new
  status = Process.run(YT_DLP_BIN, args, output: stdout, error: stderr)

  if !status.success?
    err_msg = stderr.to_s.strip
    err_msg = "Unknown Error" if err_msg.empty?
    raise "YT-DLP Error: #{err_msg}"
  end

  parsed = JSON.parse(stdout.to_s)
  if entries = parsed["entries"]?.try(&.as_a?)
    if entries.size > 0
      parsed = entries[0]
    end
  end

  {parsed, method}
end

def extract_smart(query : String)
  last_err = ""
  3.times do |i|
    attempt = i + 1
    cookie = COOKIES.get_cookie
    begin
      parsed, method = execute_yt_dlp(query, cookie, attempt)
      if parsed["formats"]?
        return {parsed, method}
      end
    rescue ex
      last_err = ex.message || "error"
      if cookie && (last_err.downcase.includes?("sign in") || last_err.downcase.includes?("bot"))
        COOKIES.report_failure(cookie)
      end
    end
    sleep 0.3
  end
  raise last_err
end

def build_smart_formats(raw_formats : Array(JSON::Any))
  best_muxed = [] of StreamInfo
  audio_only = [] of StreamInfo
  video_only = [] of StreamInfo

  raw_formats.each do |rf|
    ext = rf["ext"]?.try(&.as_s?) || ""
    proto = rf["protocol"]?.try(&.as_s?) || ""
    next if ext == "mhtml" || ext == "sb0" || ext == "sb1" || proto.starts_with?("mhtml")
    next if !proto.starts_with?("http") && !proto.starts_with?("m3u8")

    vcodec = rf["vcodec"]?.try(&.as_s?) || "none"
    acodec = rf["acodec"]?.try(&.as_s?) || "none"
    url = rf["url"]?.try(&.as_s?) || ""
    next if url.empty?

    fmt_id = rf["format_id"]?.try(&.as_s?) || "0"
    res = rf["format_note"]?.try(&.as_s?) || rf["resolution"]?.try(&.as_s?) || "unknown"

    score = 0
    score += 10 if ext.includes?("mp4") || ext.includes?("m4a")
    if tbr = rf["tbr"]?.try(&.as_f?)
      score += (tbr / 100).to_i
    end

    info = StreamInfo.new(fmt_id, ext, res, vcodec, acodec, url, score)

    if vcodec != "none" && acodec != "none"
      info.quality_score += 50 unless proto.includes?("m3u8")
      best_muxed << info
    elsif vcodec == "none" && acodec != "none"
      audio_only << info
    elsif vcodec != "none" && acodec == "none"
      video_only << info
    end
  end

  best_muxed.sort! { |a, b| b.quality_score <=> a.quality_score }
  audio_only.sort! { |a, b| b.quality_score <=> a.quality_score }
  video_only.sort! { |a, b| b.quality_score <=> a.quality_score }

  best_a = audio_only.empty? ? "" : audio_only.first.url
  best_v = best_muxed.empty? ? (video_only.empty? ? "" : video_only.first.url) : best_muxed.first.url

  {SmartFormats.new(best_muxed, audio_only, video_only), best_a, best_v}
end

def verify_auth(env)
  key = env.request.headers["X-Titan-Key"]? || env.request.headers["X-Ultra-Key"]?
  key == API_KEY
end

before_all do |env|
  env.response.headers["Access-Control-Allow-Origin"] = "*"
  env.response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
  env.response.headers["Access-Control-Allow-Headers"] = "*"
end

options "/*" do |env|
  env.response.status_code = 200
  ""
end

get "/" do |env|
  if File.exists?("index.html")
    env.response.content_type = "text/html"
    File.read("index.html")
  else
    env.response.content_type = "application/json"
    %({"system": "Crystal Fast Engine", "status": "Active"})
  end
end

get "/api/v1/extract" do |env|
  STATS.inc_total
  start_time = Time.monotonic

  if !verify_auth(env)
    STATS.inc_fail
    env.response.content_type = "application/json"
    halt env, 403, %({"success":false,"message":"Forbidden"})
  end

  url = env.params.query["url"]?
  if !url || url.empty?
    STATS.inc_fail
    env.response.content_type = "application/json"
    halt env, 400, %({"success":false,"message":"Missing URL"})
  end

  audio_only_str = env.params.query["audio_only"]? || "true"
  force_refresh_str = env.params.query["force_refresh"]? || "false"
  audio_only = audio_only_str != "false" && audio_only_str != "0"
  force_refresh = force_refresh_str == "true" || force_refresh_str == "1"

  cache_key = "#{url}_audio:#{audio_only}"

  if !force_refresh
    if cached_json = CACHE.get(cache_key)
      begin
        parsed_cache = JSON.parse(cached_json).as_h
        parsed_cache["cached"] = JSON::Any.new(true)
        parsed_cache["process_time_ms"] = JSON::Any.new((Time.monotonic - start_time).total_milliseconds)
        STATS.inc_success
        env.response.content_type = "application/json"
        halt env, 200, parsed_cache.to_json
      rescue
      end
    end
  end

  begin
    info, method = extract_smart(url)
    raw_formats = info["formats"]?.try(&.as_a?) || [] of JSON::Any
    sf, best_a, best_v = build_smart_formats(raw_formats)

    direct_url = audio_only ? (best_a.empty? ? best_v : best_a) : (best_v.empty? ? best_a : best_v)
    direct_url = info["url"]?.try(&.as_s?) || "" if direct_url.empty?

    is_live = info["is_live"]?.try(&.as_bool?) || info["was_live"]?.try(&.as_bool?) || false
    duration = (!is_live && info["duration"]?.try(&.as_f?)) ? info["duration"].as_f.to_i : 0
    vid_id = info["id"]?.try(&.as_s?) || ""
    title = info["title"]?.try(&.as_s?) || "Unknown"

    thumbs = [] of ThumbnailModel
    if raw_thumbs = info["thumbnails"]?.try(&.as_a?)
      raw_thumbs.each do |rt|
        u = rt["url"]?.try(&.as_s?) || ""
        w = rt["width"]?.try(&.as_i?) || 0
        h = rt["height"]?.try(&.as_i?) || 0
        thumbs << ThumbnailModel.new(u, w, h)
      end
    end

    resp = MediaResponse.new(
      true,
      (Time.monotonic - start_time).total_milliseconds,
      false,
      method,
      vid_id,
      title,
      duration,
      is_live,
      thumbs,
      direct_url,
      sf,
      raw_formats.size
    )

    resp_json = resp.to_json
    CACHE.set(cache_key, resp_json)
    STATS.inc_success
    
    env.response.content_type = "application/json"
    resp_json
  rescue ex
    STATS.inc_fail
    env.response.content_type = "application/json"
    halt env, 500, %({"success":false,"message":#{ex.message.to_json},"process_time_ms":#{(Time.monotonic - start_time).total_milliseconds}})
  end
end

ws "/api/v1/ws/stream" do |socket|
  unless CONFIG.allow_new_ws_connections
    socket.close
  end

  authenticated = false

  socket.on_message do |msg|
    begin
      req = JSON.parse(msg)
      
      if !authenticated
        if req["auth"]?.try(&.as_s?) == API_KEY
          authenticated = true
          STATS.inc_ws
        else
          socket.close
        end
        next
      end

      url = req["url"]?.try(&.as_s?)
      if !url || url.empty?
        socket.send(%({"success":false,"message":"Missing URL"}))
        next
      end

      audio_only = req["audio_only"]?.try(&.as_bool?)
      audio_only = true if audio_only.nil?

      STATS.inc_total
      start_time = Time.monotonic
      cache_key = "#{url}_audio:#{audio_only}"

      if cached_json = CACHE.get(cache_key)
        parsed_cache = JSON.parse(cached_json).as_h
        parsed_cache["cached"] = JSON::Any.new(true)
        parsed_cache["process_time_ms"] = JSON::Any.new((Time.monotonic - start_time).total_milliseconds)
        STATS.inc_success
        socket.send(parsed_cache.to_json)
        next
      end

      info, method = extract_smart(url)
      raw_formats = info["formats"]?.try(&.as_a?) || [] of JSON::Any
      sf, best_a, best_v = build_smart_formats(raw_formats)

      direct_url = audio_only ? (best_a.empty? ? best_v : best_a) : (best_v.empty? ? best_a : best_v)
      direct_url = info["url"]?.try(&.as_s?) || "" if direct_url.empty?

      is_live = info["is_live"]?.try(&.as_bool?) || info["was_live"]?.try(&.as_bool?) || false
      duration = (!is_live && info["duration"]?.try(&.as_f?)) ? info["duration"].as_f.to_i : 0
      vid_id = info["id"]?.try(&.as_s?) || ""
      title = info["title"]?.try(&.as_s?) || "Unknown"

      resp = MediaResponse.new(
        true,
        (Time.monotonic - start_time).total_milliseconds,
        false,
        method,
        vid_id,
        title,
        duration,
        is_live,
        [] of ThumbnailModel,
        direct_url,
        sf,
        raw_formats.size
      )

      resp_json = resp.to_json
      CACHE.set(cache_key, resp_json)
      STATS.inc_success
      socket.send(resp_json)
    rescue ex
      STATS.inc_fail
      socket.send(%({"success":false,"message":#{ex.message.to_json},"process_time_ms":#{(Time.monotonic - start_time).total_milliseconds}}))
    end
  end

  socket.on_close do
    STATS.dec_ws if authenticated
  end
end

get "/api/v1/admin/stats" do |env|
  if !verify_auth(env)
    env.response.content_type = "application/json"
    halt env, 403, %({"success":false,"message":"Forbidden"})
  end
  env.response.content_type = "application/json"
  {
    status: "online",
    uptime_seconds: (Time.utc - SERVER_START).total_seconds,
    total_requests: STATS.total_requests,
    successful_requests: STATS.successful_requests,
    failed_requests: STATS.failed_requests,
    active_ws_connections: STATS.active_ws_connections,
    cached_files_count: CACHE.size,
    memory_usage: get_memory_usage,
    available_cookies: COOKIES.pool.size
  }.to_json
end

post "/api/v1/admin/clear_cache" do |env|
  if !verify_auth(env)
    env.response.content_type = "application/json"
    halt env, 403, %({"success":false,"message":"Forbidden"})
  end
  CACHE.clear_all
  env.response.content_type = "application/json"
  %({"success": true, "message": "Cache completely cleared."})
end

post "/api/v1/admin/config" do |env|
  if !verify_auth(env)
    env.response.content_type = "application/json"
    halt env, 403, %({"success":false,"message":"Forbidden"})
  end
  begin
    body = env.request.body.not_nil!.gets_to_end
    req = JSON.parse(body)
    smart = req["smart_validation_enabled"]?.try(&.as_bool?)
    ws = req["allow_new_ws_connections"]?.try(&.as_bool?)
    CONFIG.update(smart, ws)
  rescue
  end
  env.response.content_type = "application/json"
  {
    success: true,
    current_config: {
      smart_validation_enabled: CONFIG.smart_validation_enabled,
      allow_new_ws_connections: CONFIG.allow_new_ws_connections
    }
  }.to_json
end

spawn do
  loop do
    sleep 2.minutes
    CACHE.clean
    COOKIES.refresh
  end
end

port = ENV.fetch("PORT", "8080").to_i
Kemal.run do |config|
  config.server.bind_tcp("0.0.0.0", port)
end

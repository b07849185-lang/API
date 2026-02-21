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

alias LogEntry = {
  time: String, 
  ip: String, 
  country: String, 
  url: String, 
  success: Bool, 
  process_time_ms: Float64
}

class RequestLogger
  @logs = Array(LogEntry).new
  @mutex = Mutex.new

  def add(ip : String, country : String, url : String, success : Bool, process_time : Float64)
    @mutex.synchronize do
      @logs.unshift({
        time: Time.utc.to_s, 
        ip: ip, 
        country: country, 
        url: url, 
        success: success, 
        process_time_ms: process_time
      })
      if @logs.size > 200
        @logs.pop
      end
    end
  end

  def get_logs
    @mutex.synchronize { @logs.dup }
  end
end

LOGS = RequestLogger.new

class DynamicConfig
  property api_enabled : Bool = true
  property smart_validation_enabled : Bool = true
  property allow_new_ws_connections : Bool = true
  @mutex = Mutex.new

  def update(api : Bool?, smart : Bool?, ws : Bool?)
    @mutex.synchronize do
      @api_enabled = api unless api.nil?
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
      if candidates.empty?
        candidates = @pool
      end
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
    @mutex.synchronize do
      @store.clear
    end
  end

  def size
    @mutex.synchronize do
      @store.size
    end
  end
end

CACHE = MemoryCache.new

def execute_yt_dlp(query : String, cookie : String?, attempt : Int32)
  args = [
    "--dump-json", 
    "--no-warnings", 
    "--skip-download", 
    "--no-playlist", 
    "--socket-timeout", "10", 
    "--compat-options", "no-youtube-unavailable-videos", 
    "--geo-bypass", 
    "--impersonate", "chrome"
  ]
  
  method = ""

  if cookie && !cookie.empty?
    args << "--cookies"
    args << cookie
    if attempt == 1
      args << "--extractor-args"
      args << "youtube:player_client=web;player_skip=configs"
      args << "--remote-components"
      args << "ejs:github,ejs:npm"
      method = "Titan Engine (Web + Cookie + EJS Fast)"
    else
      args << "--extractor-args"
      args << "youtube:player_client=mweb;player_skip=configs"
      args << "--remote-components"
      args << "ejs:github,ejs:npm"
      method = "Titan Engine (Mobile Web + Cookie + EJS)"
    end
  else
    if attempt == 1
      args << "--extractor-args"
      args << "youtube:player_client=android,ios;player_skip=webpage,configs"
      method = "Titan Engine (Android API - Lightning)"
    else
      args << "--extractor-args"
      args << "youtube:player_client=ios,android;player_skip=webpage,configs"
      method = "Titan Engine (iOS API - Lightning)"
    end
  end

  if !query.starts_with?("http") && !query.starts_with?("ytsearch")
    args << "ytsearch1:#{query}"
  else
    args << query
  end

  stdout = IO::Memory.new
  stderr = IO::Memory.new
  status = Process.run(YT_DLP_BIN, args, output: stdout, error: stderr)

  if !status.success?
    err_msg = stderr.to_s.strip
    if err_msg.empty?
      err_msg = "Unknown Error"
    end
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
  2.times do |i|
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
    sleep(300.milliseconds)
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
    
    if ext == "mhtml" || ext == "sb0" || ext == "sb1" || proto.starts_with?("mhtml")
      next
    end
    
    if !proto.starts_with?("http") && !proto.starts_with?("m3u8")
      next
    end

    vcodec = rf["vcodec"]?.try(&.as_s?) || "none"
    acodec = rf["acodec"]?.try(&.as_s?) || "none"
    url = rf["url"]?.try(&.as_s?) || ""
    
    if url.empty?
      next
    end

    fmt_id = rf["format_id"]?.try(&.as_s?) || "0"
    res = rf["format_note"]?.try(&.as_s?) || rf["resolution"]?.try(&.as_s?) || "unknown"

    score = 0
    if ext.includes?("mp4") || ext.includes?("m4a")
      score += 10
    end
    
    if res_match = res.match(/\d+/)
      score += res_match[0].to_i
    end
    
    if tbr = rf["tbr"]?.try(&.as_f?)
      score += (tbr / 100).to_i
    end

    info = StreamInfo.new(fmt_id, ext, res, vcodec, acodec, url, score)

    if vcodec != "none" && acodec != "none"
      if !proto.includes?("m3u8")
        info.quality_score += 5000
      end
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

  best_a = ""
  if !audio_only.empty?
    best_a = audio_only.first.url
  end
  
  best_v = ""
  if !best_muxed.empty?
    best_v = best_muxed.first.url
  elsif !video_only.empty?
    best_v = video_only.first.url
  end

  {SmartFormats.new(best_muxed, audio_only, video_only), best_a, best_v}
end

def verify_auth(env)
  key = env.request.headers["X-Titan-Key"]? || env.request.headers["X-Ultra-Key"]?
  if key == API_KEY
    return true
  end
  return false
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
  start_time = Time.utc
  
  ip = env.request.headers["Fly-Client-IP"]? || env.request.headers["X-Forwarded-For"]? || env.request.remote_address.try(&.to_s) || "Unknown"
  country = env.request.headers["Fly-Region"]? || env.request.headers["CF-IPCountry"]? || "Unknown"

  if !CONFIG.api_enabled
    STATS.inc_fail
    LOGS.add(ip, country, env.params.query["url"]?.to_s, false, 0.0)
    env.response.content_type = "application/json"
    halt env, 503, %({"success":false,"message":"API Locked by Administrator."})
  end

  if !verify_auth(env)
    STATS.inc_fail
    LOGS.add(ip, country, env.params.query["url"]?.to_s, false, 0.0)
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
        process_time = (Time.utc - start_time).total_milliseconds
        parsed_cache["process_time_ms"] = JSON::Any.new(process_time)
        STATS.inc_success
        LOGS.add(ip, country, url, true, process_time)
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

    direct_url = ""
    if audio_only
      if best_a.empty?
        direct_url = best_v
      else
        direct_url = best_a
      end
    else
      if best_v.empty?
        direct_url = best_a
      else
        direct_url = best_v
      end
    end

    if direct_url.empty?
      direct_url = info["url"]?.try(&.as_s?) || ""
    end

    is_live = false
    if info["is_live"]?.try(&.as_bool?) || info["was_live"]?.try(&.as_bool?)
      is_live = true
    end
    
    duration = 0
    if !is_live && info["duration"]?.try(&.as_f?)
      duration = info["duration"].as_f.to_i
    end
    
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

    process_time = (Time.utc - start_time).total_milliseconds

    resp = MediaResponse.new(
      true,
      process_time,
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
    LOGS.add(ip, country, url, true, process_time)
    
    env.response.content_type = "application/json"
    resp_json
  rescue ex
    STATS.inc_fail
    process_time = (Time.utc - start_time).total_milliseconds
    LOGS.add(ip, country, url.to_s, false, process_time)
    env.response.content_type = "application/json"
    halt env, 500, %({"success":false,"message":#{ex.message.to_json},"process_time_ms":#{process_time}})
  end
end

ws "/api/v1/ws/stream" do |socket|
  if !CONFIG.allow_new_ws_connections || !CONFIG.api_enabled
    socket.close
  end

  authenticated = false

  socket.on_message do |msg|
    start_time = Time.utc
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
      if audio_only.nil?
        audio_only = true
      end

      STATS.inc_total
      cache_key = "#{url}_audio:#{audio_only}"

      if cached_json = CACHE.get(cache_key)
        parsed_cache = JSON.parse(cached_json).as_h
        parsed_cache["cached"] = JSON::Any.new(true)
        parsed_cache["process_time_ms"] = JSON::Any.new((Time.utc - start_time).total_milliseconds)
        STATS.inc_success
        socket.send(parsed_cache.to_json)
        next
      end

      info, method = extract_smart(url)
      raw_formats = info["formats"]?.try(&.as_a?) || [] of JSON::Any
      sf, best_a, best_v = build_smart_formats(raw_formats)

      direct_url = ""
      if audio_only
        if best_a.empty?
          direct_url = best_v
        else
          direct_url = best_a
        end
      else
        if best_v.empty?
          direct_url = best_a
        else
          direct_url = best_v
        end
      end

      if direct_url.empty?
        direct_url = info["url"]?.try(&.as_s?) || ""
      end

      is_live = false
      if info["is_live"]?.try(&.as_bool?) || info["was_live"]?.try(&.as_bool?)
        is_live = true
      end
      
      duration = 0
      if !is_live && info["duration"]?.try(&.as_f?)
        duration = info["duration"].as_f.to_i
      end
      
      vid_id = info["id"]?.try(&.as_s?) || ""
      title = info["title"]?.try(&.as_s?) || "Unknown"

      resp = MediaResponse.new(
        true,
        (Time.utc - start_time).total_milliseconds,
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
      socket.send(%({"success":false,"message":#{ex.message.to_json},"process_time_ms":#{(Time.utc - start_time).total_milliseconds}}))
    end
  end

  socket.on_close do
    if authenticated
      STATS.dec_ws
    end
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
    api_enabled: CONFIG.api_enabled,
    uptime_seconds: (Time.utc - SERVER_START).total_seconds,
    total_requests: STATS.total_requests,
    successful_requests: STATS.successful_requests,
    failed_requests: STATS.failed_requests,
    active_ws_connections: STATS.active_ws_connections,
    cached_files_count: CACHE.size,
    memory_usage: get_memory_usage,
    available_cookies: COOKIES.pool.size,
    recent_logs: LOGS.get_logs
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
    api_status = req["api_enabled"]?.try(&.as_bool?)
    CONFIG.update(api_status, smart, ws)
  rescue
  end
  env.response.content_type = "application/json"
  {
    success: true,
    current_config: {
      api_enabled: CONFIG.api_enabled,
      smart_validation_enabled: CONFIG.smart_validation_enabled,
      allow_new_ws_connections: CONFIG.allow_new_ws_connections
    }
  }.to_json
end

spawn do
  loop do
    sleep(120.seconds)
    CACHE.clean
    COOKIES.refresh
  end
end

Kemal.config.host_binding = "0.0.0.0"
Kemal.config.port = ENV.fetch("PORT", "8080").to_i
Kemal.run

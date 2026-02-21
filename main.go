package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math/rand"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"
)

var (
	API_KEY           = getEnv("TITAN_SECRET_KEY", "Titan_2026_Ultra_Fast")
	COOKIE_DIR        = getEnv("TITAN_COOKIE_DIR", "cookies")
	CACHE_TTL         = getEnvInt("CACHE_TTL", 14400)
	COOKIE_BAN_TIME   = int64(getEnvInt("COOKIE_BAN_TIME", 3600))
	IMPERSONATE_TARGET= getEnv("YT_IMPERSONATE_TARGET", "chrome")
	YT_DLP_BIN        = getEnv("YT_DLP_BIN", "yt-dlp")
)

type APIStats struct {
	TotalRequests         int64
	SuccessfulRequests    int64
	FailedRequests        int64
	ActiveWsConnections   int64
	ServerStartTime       time.Time
	mu                    sync.RWMutex
}

var stats = &APIStats{ServerStartTime: time.Now()}

type DynamicConfig struct {
	SmartValidationEnabled    bool
	ValidationIntervalSeconds int
	AllowNewWsConnections     bool
	mu                        sync.RWMutex
}

var config = &DynamicConfig{
	SmartValidationEnabled:    true,
	ValidationIntervalSeconds: 7200,
	AllowNewWsConnections:     true,
}

type StreamInfo struct {
	FormatID     string `json:"format_id"`
	Ext          string `json:"ext"`
	Resolution   string `json:"resolution"`
	Vcodec       string `json:"vcodec"`
	Acodec       string `json:"acodec"`
	URL          string `json:"url"`
	QualityScore int    `json:"quality_score"`
}

type SmartFormats struct {
	BestMuxed []StreamInfo `json:"best_muxed"`
	AudioOnly []StreamInfo `json:"audio_only"`
	VideoOnly []StreamInfo `json:"video_only"`
}

type ThumbnailModel struct {
	URL    string `json:"url"`
	Width  int    `json:"width"`
	Height int    `json:"height"`
}

type MediaResponse struct {
	Success          bool             `json:"success"`
	ProcessTimeMs    float64          `json:"process_time_ms"`
	Cached           bool             `json:"cached"`
	ExtractionMethod string           `json:"extraction_method"`
	VideoID          string           `json:"video_id"`
	Title            string           `json:"title"`
	Duration         int              `json:"duration"`
	IsLive           bool             `json:"is_live"`
	Thumbnails       []ThumbnailModel `json:"thumbnails"`
	DirectStreamURL  string           `json:"direct_stream_url"`
	SmartFormats     SmartFormats     `json:"smart_formats"`
	RawFallbackCount int              `json:"raw_fallback_count"`
}

type ErrorResponse struct {
	Success       bool    `json:"success"`
	ErrorCode     int     `json:"error_code"`
	Message       string  `json:"message"`
	ProcessTimeMs float64 `json:"process_time_ms"`
}

type CacheItem struct {
	Timestamp int64
	Data      MediaResponse
}

type MemoryCache struct {
	items map[string]CacheItem
	mu    sync.RWMutex
}

var cacheEngine = &MemoryCache{items: make(map[string]CacheItem)}

func (c *MemoryCache) Get(key string) (MediaResponse, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	item, found := c.items[key]
	if !found {
		return MediaResponse{}, false
	}
	if time.Now().Unix()-item.Timestamp < int64(CACHE_TTL) {
		return item.Data, true
	}
	return MediaResponse{}, false
}

func (c *MemoryCache) Set(key string, data MediaResponse) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.items[key] = CacheItem{Timestamp: time.Now().Unix(), Data: data}
}

func (c *MemoryCache) Remove(key string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	delete(c.items, key)
}

func (c *MemoryCache) ClearAll() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.items = make(map[string]CacheItem)
}

func (c *MemoryCache) Cleanup() {
	c.mu.Lock()
	defer c.mu.Unlock()
	now := time.Now().Unix()
	for k, v := range c.items {
		if now-v.Timestamp >= int64(CACHE_TTL) {
			delete(c.items, k)
		}
	}
}

type EnterpriseCookieManager struct {
	Directory  string
	Pool       []string
	Banned     map[string]int64
	LastUsed   map[string]int64
	ReuseDelay int64
	mu         sync.Mutex
}

var cookieVault = &EnterpriseCookieManager{
	Directory:  COOKIE_DIR,
	Banned:     make(map[string]int64),
	LastUsed:   make(map[string]int64),
	ReuseDelay: 2,
}

func init() {
	os.MkdirAll(COOKIE_DIR, os.ModePerm)
	cookieVault.RefreshPoolSync()
}

func (cm *EnterpriseCookieManager) RefreshPoolSync() {
	cm.mu.Lock()
	defer cm.mu.Unlock()
	files, _ := filepath.Glob(filepath.Join(cm.Directory, "*.txt"))
	now := time.Now().Unix()
	for k, v := range cm.Banned {
		if now-v > COOKIE_BAN_TIME {
			delete(cm.Banned, k)
		}
	}
	var newPool []string
	for _, f := range files {
		if _, isBanned := cm.Banned[f]; !isBanned {
			if info, err := os.Stat(f); err == nil && info.Size() > 0 {
				newPool = append(newPool, f)
			}
		}
	}
	cm.Pool = newPool
}

func (cm *EnterpriseCookieManager) GetCookie() string {
	cm.mu.Lock()
	defer cm.mu.Unlock()
	if len(cm.Pool) == 0 {
		return ""
	}
	now := time.Now().Unix()
	var candidates []string
	for _, c := range cm.Pool {
		if last, ok := cm.LastUsed[c]; !ok || now-last > cm.ReuseDelay {
			candidates = append(candidates, c)
		}
	}
	if len(candidates) == 0 {
		candidates = cm.Pool
	}
	chosen := candidates[rand.Intn(len(candidates))]
	cm.LastUsed[chosen] = now
	return chosen
}

func (cm *EnterpriseCookieManager) ReportFailure(cookie string) {
	if cookie == "" {
		return
	}
	cm.mu.Lock()
	defer cm.mu.Unlock()
	cm.Banned[cookie] = time.Now().Unix()
	for i, c := range cm.Pool {
		if c == cookie {
			cm.Pool = append(cm.Pool[:i], cm.Pool[i+1:]...)
			break
		}
	}
}

func execCmd(timeout time.Duration, name string, args ...string) ([]byte, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, name, args...)
	var out bytes.Buffer
	cmd.Stdout = &out
	err := cmd.Run()
	if ctx.Err() == context.DeadlineExceeded {
		return nil, fmt.Errorf("timeout")
	}
	return out.Bytes(), err
}

func extractSmart(rawQuery string) (map[string]interface{}, string, error) {
	query := strings.TrimSpace(rawQuery)
	if !strings.HasPrefix(strings.ToLower(query), "http") && !strings.HasPrefix(strings.ToLower(query), "ytsearch") {
		query = "ytsearch1:" + query
	}

	var lastErr error
	for attempt := 1; attempt <= 3; attempt++ {
		cookie := cookieVault.GetCookie()
		methodUsed := ""
		args := []string{"--dump-json", "--no-warnings", "--skip-download", "--no-playlist", "--default-search", "ytsearch"}
		
		if cookie != "" {
			args = append(args, "--cookies", cookie)
		}

		if attempt == 1 {
			methodUsed = "CLI (Web Client + EJS)"
			args = append(args, "--extractor-args", "youtube:player_client=web", "--remote-components", "ejs:github", "--socket-timeout", "8")
		} else if attempt == 2 {
			methodUsed = "CLI (Android Fallback)"
			args = append(args, "--extractor-args", "youtube:player_client=android;player_skip=configs", "--socket-timeout", "10")
		} else {
			methodUsed = "CLI (Web Last Resort)"
			args = append(args, "--extractor-args", "youtube:player_client=web", "--socket-timeout", "12")
		}
		
		args = append(args, query)

		out, err := execCmd(15*time.Second, YT_DLP_BIN, args...)
		if err == nil && len(out) > 0 {
			var info map[string]interface{}
			lines := strings.Split(string(out), "\n")
			for _, line := range lines {
				if strings.HasPrefix(strings.TrimSpace(line), "{") {
					if json.Unmarshal([]byte(line), &info) == nil {
						if entries, ok := info["entries"].([]interface{}); ok && len(entries) > 0 {
							if firstEntry, ok2 := entries[0].(map[string]interface{}); ok2 {
								info = firstEntry
							}
						}
						if _, ok := info["formats"]; ok {
							return info, methodUsed, nil
						}
					}
				}
			}
		}
		
		lastErr = fmt.Errorf("attempt %d failed", attempt)
		if cookie != "" && err != nil && (strings.Contains(strings.ToLower(err.Error()), "sign in") || strings.Contains(strings.ToLower(err.Error()), "bot")) {
			cookieVault.ReportFailure(cookie)
		}
		time.Sleep(500 * time.Millisecond)
	}
	return nil, "", fmt.Errorf("Extraction failed: %v", lastErr)
}

func buildSmartFormats(formatsList []interface{}) (SmartFormats, string, string) {
	sf := SmartFormats{}
	var bestA, bestV string

	for _, fRaw := range formatsList {
		f, ok := fRaw.(map[string]interface{})
		if !ok {
			continue
		}

		ext, _ := f["ext"].(string)
		if ext == "mhtml" || ext == "sb0" || ext == "sb1" {
			continue
		}

		proto, _ := f["protocol"].(string)
		if !strings.HasPrefix(proto, "http") && !strings.HasPrefix(proto, "m3u8") {
			continue
		}

		vcodec, _ := f["vcodec"].(string)
		if vcodec == "" { vcodec = "none" }
		acodec, _ := f["acodec"].(string)
		if acodec == "" { acodec = "none" }
		urlStr, _ := f["url"].(string)
		if urlStr == "" {
			continue
		}

		fmtID, _ := f["format_id"].(string)
		if fmtID == "" { fmtID = "0" }
		
		res := "unknown"
		if note, ok := f["format_note"].(string); ok && note != "" {
			res = note
		} else if resRaw, ok := f["resolution"].(string); ok {
			res = resRaw
		}

		tbr, _ := f["tbr"].(float64)

		score := 0
		if strings.Contains(ext, "mp4") || strings.Contains(ext, "m4a") {
			score += 10
		}
		score += int(tbr / 100)

		info := StreamInfo{
			FormatID:     fmtID,
			Ext:          ext,
			Resolution:   res,
			Vcodec:       vcodec,
			Acodec:       acodec,
			URL:          urlStr,
			QualityScore: score,
		}

		if vcodec != "none" && acodec != "none" {
			if !strings.Contains(proto, "m3u8") {
				info.QualityScore += 50
			}
			sf.BestMuxed = append(sf.BestMuxed, info)
		} else if vcodec == "none" && acodec != "none" {
			sf.AudioOnly = append(sf.AudioOnly, info)
		} else if vcodec != "none" && acodec == "none" {
			sf.VideoOnly = append(sf.VideoOnly, info)
		}
	}

	sort.Slice(sf.BestMuxed, func(i, j int) bool { return sf.BestMuxed[i].QualityScore > sf.BestMuxed[j].QualityScore })
	sort.Slice(sf.AudioOnly, func(i, j int) bool { return sf.AudioOnly[i].QualityScore > sf.AudioOnly[j].QualityScore })
	sort.Slice(sf.VideoOnly, func(i, j int) bool { return sf.VideoOnly[i].QualityScore > sf.VideoOnly[j].QualityScore })

	if len(sf.AudioOnly) > 0 { bestA = sf.AudioOnly[0].URL }
	if len(sf.BestMuxed) > 0 { 
		bestV = sf.BestMuxed[0].URL 
	} else if len(sf.VideoOnly) > 0 {
		bestV = sf.VideoOnly[0].URL
	}

	return sf, bestA, bestV
}

func verifyAuth(r *http.Request) bool {
	key := r.Header.Get("X-Titan-Key")
	return key == API_KEY
}

func getMemoryUsage() string {
	return "Available via OS tools in Go" 
}

func extractHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	if !verifyAuth(r) {
		w.WriteHeader(http.StatusForbidden)
		json.NewEncoder(w).Encode(map[string]string{"message": "Forbidden"})
		return
	}

	stats.mu.Lock()
	stats.TotalRequests++
	stats.mu.Unlock()

	start := time.Now()
	urlParam := r.URL.Query().Get("url")
	audioOnlyStr := r.URL.Query().Get("audio_only")
	audioOnly := audioOnlyStr == "" || audioOnlyStr == "true" || audioOnlyStr == "1"
	forceRefreshStr := r.URL.Query().Get("force_refresh")
	forceRefresh := forceRefreshStr == "true" || forceRefreshStr == "1"

	if urlParam == "" {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(ErrorResponse{Success: false, Message: "Missing URL"})
		return
	}

	cacheKey := fmt.Sprintf("%s_audio:%v", urlParam, audioOnly)

	if !forceRefresh {
		if cached, found := cacheEngine.Get(cacheKey); found {
			cached.Cached = true
			cached.ProcessTimeMs = float64(time.Since(start).Milliseconds())
			stats.mu.Lock()
			stats.SuccessfulRequests++
			stats.mu.Unlock()
			json.NewEncoder(w).Encode(cached)
			return
		}
	}

	info, method, err := extractSmart(urlParam)
	if err != nil {
		stats.mu.Lock()
		stats.FailedRequests++
		stats.mu.Unlock()
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(ErrorResponse{Success: false, Message: err.Error(), ProcessTimeMs: float64(time.Since(start).Milliseconds())})
		return
	}

	var rawFormats []interface{}
	if f, ok := info["formats"].([]interface{}); ok {
		rawFormats = f
	}

	smartFormats, bestA, bestV := buildSmartFormats(rawFormats)

	directURL := ""
	if audioOnly {
		if bestA != "" { directURL = bestA } else { directURL = bestV }
	} else {
		if bestV != "" { directURL = bestV } else { directURL = bestA }
	}

	if directURL == "" {
		if u, ok := info["url"].(string); ok { directURL = u }
	}

	if directURL == "" {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(ErrorResponse{Success: false, Message: "No valid streams found"})
		return
	}

	dur := 0
	isLive := false
	if l, ok := info["is_live"].(bool); ok && l { isLive = true }
	if !isLive {
		if d, ok := info["duration"].(float64); ok { dur = int(d) }
	}

	vidID, _ := info["id"].(string)
	title, _ := info["title"].(string)
	if title == "" { title = "Unknown" }

	resp := MediaResponse{
		Success:          true,
		ProcessTimeMs:    float64(time.Since(start).Milliseconds()),
		Cached:           false,
		ExtractionMethod: method,
		VideoID:          vidID,
		Title:            title,
		Duration:         dur,
		IsLive:           isLive,
		DirectStreamURL:  directURL,
		SmartFormats:     smartFormats,
		RawFallbackCount: len(rawFormats),
	}

	cacheEngine.Set(cacheKey, resp)
	stats.mu.Lock()
	stats.SuccessfulRequests++
	stats.mu.Unlock()

	json.NewEncoder(w).Encode(resp)
}

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool { return true },
}

func wsHandler(w http.ResponseWriter, r *http.Request) {
	config.mu.RLock()
	allow := config.AllowNewWsConnections
	config.mu.RUnlock()

	if !allow {
		http.Error(w, "New connections disabled", http.StatusServiceUnavailable)
		return
	}

	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		return
	}
	defer conn.Close()

	// Auth step
	var authMsg map[string]interface{}
	conn.SetReadDeadline(time.Now().Add(5 * time.Second))
	err = conn.ReadJSON(&authMsg)
	if err != nil || authMsg["auth"] != API_KEY {
		conn.WriteMessage(websocket.CloseMessage, websocket.FormatCloseMessage(1008, "Unauthorized"))
		return
	}
	conn.SetReadDeadline(time.Time{}) // Reset deadline

	stats.mu.Lock()
	stats.ActiveWsConnections++
	stats.mu.Unlock()

	defer func() {
		stats.mu.Lock()
		stats.ActiveWsConnections--
		stats.mu.Unlock()
	}()

	for {
		var req map[string]interface{}
		err := conn.ReadJSON(&req)
		if err != nil {
			break
		}

		url, _ := req["url"].(string)
		audioOnly := true
		if ao, ok := req["audio_only"].(bool); ok {
			audioOnly = ao
		}

		if url == "" {
			conn.WriteJSON(map[string]interface{}{"success": false, "message": "Missing URL"})
			continue
		}

		start := time.Now()
		cacheKey := fmt.Sprintf("%s_audio:%v", url, audioOnly)
		
		stats.mu.Lock()
		stats.TotalRequests++
		stats.mu.Unlock()

		if cached, found := cacheEngine.Get(cacheKey); found {
			cached.Cached = true
			cached.ProcessTimeMs = float64(time.Since(start).Milliseconds())
			stats.mu.Lock()
			stats.SuccessfulRequests++
			stats.mu.Unlock()
			conn.WriteJSON(cached)
			continue
		}

		info, method, err := extractSmart(url)
		if err != nil {
			stats.mu.Lock()
			stats.FailedRequests++
			stats.mu.Unlock()
			conn.WriteJSON(ErrorResponse{Success: false, Message: err.Error(), ProcessTimeMs: float64(time.Since(start).Milliseconds())})
			continue
		}

		var rawFormats []interface{}
		if f, ok := info["formats"].([]interface{}); ok { rawFormats = f }
		smartFormats, bestA, bestV := buildSmartFormats(rawFormats)

		directURL := ""
		if audioOnly {
			if bestA != "" { directURL = bestA } else { directURL = bestV }
		} else {
			if bestV != "" { directURL = bestV } else { directURL = bestA }
		}
		if directURL == "" { if u, ok := info["url"].(string); ok { directURL = u } }

		dur := 0
		isLive := false
		if l, ok := info["is_live"].(bool); ok && l { isLive = true }
		if !isLive { if d, ok := info["duration"].(float64); ok { dur = int(d) } }

		vidID, _ := info["id"].(string)
		title, _ := info["title"].(string)
		if title == "" { title = "Unknown" }

		resp := MediaResponse{
			Success:          true,
			ProcessTimeMs:    float64(time.Since(start).Milliseconds()),
			Cached:           false,
			ExtractionMethod: method,
			VideoID:          vidID,
			Title:            title,
			Duration:         dur,
			IsLive:           isLive,
			DirectStreamURL:  directURL,
			SmartFormats:     smartFormats,
		}

		cacheEngine.Set(cacheKey, resp)
		stats.mu.Lock()
		stats.SuccessfulRequests++
		stats.mu.Unlock()

		conn.WriteJSON(resp)
	}
}

func rootHandler(w http.ResponseWriter, r *http.Request) {
	if _, err := os.Stat("index.html"); err == nil {
		http.ServeFile(w, r, "index.html")
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"system": "TitanOS Core", "status": "Active", "engine": "Go Native Engine"})
}

func backgroundTasks() {
	for {
		time.Sleep(120 * time.Second)
		cacheEngine.Cleanup()
		cookieVault.RefreshPoolSync()
	}
}

func getEnv(key, fallback string) string {
	if value, exists := os.LookupEnv(key); exists {
		return value
	}
	return fallback
}

func getEnvInt(key string, fallback int) int {
	if valueStr, exists := os.LookupEnv(key); exists {
		if value, err := strconv.Atoi(valueStr); err == nil {
			return value
		}
	}
	return fallback
}

func main() {
	go backgroundTasks()

	http.HandleFunc("/", rootHandler)
	http.HandleFunc("/api/v1/extract", extractHandler)
	http.HandleFunc("/api/v1/ws/stream", wsHandler)

	port := getEnv("PORT", "8080")
	fmt.Printf("Starting TitanOS Go Engine on port %s...\n", port)
	log.Fatal(http.ListenAndServe("0.0.0.0:"+port, nil))
}

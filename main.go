package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math/rand"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gorilla/websocket"
)

var (
	APIKey         = getEnv("TITAN_SECRET_KEY", "Titan_2026_Ultra_Fast")
	CookieDir      = getEnv("TITAN_COOKIE_DIR", "cookies")
	CacheTTL       = 14400.0
	CookieBanTime  = 3600.0
	YtDlpBin       = getEnv("YT_DLP_BIN", "yt-dlp")
	ServerStart    = time.Now()
)

var Stats = struct {
	TotalRequests      uint64
	SuccessfulRequests uint64
	FailedRequests     uint64
	ActiveWS           int64
}{}

var DynamicConfig = struct {
	sync.RWMutex
	SmartValidation bool
	AllowNewWS      bool
}{SmartValidation: true, AllowNewWS: true}

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
	ProcessTime      float64          `json:"process_time_ms"`
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

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool { return true },
}

func getEnv(key, fallback string) string {
	if value, exists := os.LookupEnv(key); exists {
		return value
	}
	return fallback
}

func getMemoryUsage() string {
	data, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return "Unknown"
	}
	lines := strings.Split(string(data), "\n")
	var total, available int
	for _, line := range lines {
		if strings.HasPrefix(line, "MemTotal:") {
			fmt.Sscanf(line, "MemTotal: %d kB", &total)
		} else if strings.HasPrefix(line, "MemAvailable:") {
			fmt.Sscanf(line, "MemAvailable: %d kB", &available)
		}
	}
	if total > 0 {
		used := total - available
		return fmt.Sprintf("%.2f MB / %.2f MB", float64(used)/1024.0, float64(total)/1024.0)
	}
	return "Unknown"
}

type CookieManager struct {
	sync.Mutex
	dir        string
	pool       []string
	banned     map[string]time.Time
	lastUsed   map[string]time.Time
}

func NewCookieManager(dir string) *CookieManager {
	os.MkdirAll(dir, os.ModePerm)
	cm := &CookieManager{
		dir:      dir,
		banned:   make(map[string]time.Time),
		lastUsed: make(map[string]time.Time),
	}
	cm.Refresh()
	return cm
}

func (cm *CookieManager) Refresh() {
	cm.Lock()
	defer cm.Unlock()
	now := time.Now()
	for k, v := range cm.banned {
		if now.Sub(v).Seconds() > CookieBanTime {
			delete(cm.banned, k)
		}
	}
	files, _ := filepath.Glob(filepath.Join(cm.dir, "*.txt"))
	cm.pool = []string{}
	for _, f := range files {
		if info, err := os.Stat(f); err == nil && info.Size() > 0 {
			if _, banned := cm.banned[f]; !banned {
				cm.pool = append(cm.pool, f)
			}
		}
	}
}

func (cm *CookieManager) GetCookie() string {
	cm.Lock()
	defer cm.Unlock()
	if len(cm.pool) == 0 {
		return ""
	}
	now := time.Now()
	var candidates []string
	for _, c := range cm.pool {
		if now.Sub(cm.lastUsed[c]).Seconds() > 2 {
			candidates = append(candidates, c)
		}
	}
	if len(candidates) == 0 {
		candidates = cm.pool
	}
	chosen := candidates[rand.Intn(len(candidates))]
	cm.lastUsed[chosen] = now
	return chosen
}

func (cm *CookieManager) ReportFailure(cookie string) {
	if cookie == "" {
		return
	}
	cm.Lock()
	defer cm.Unlock()
	cm.banned[cookie] = time.Now()
	var newPool []string
	for _, c := range cm.pool {
		if c != cookie {
			newPool = append(newPool, c)
		}
	}
	cm.pool = newPool
}

var cookieVault = NewCookieManager(CookieDir)

type CacheItem struct {
	Timestamp time.Time
	Data      []byte
}

type MemoryCache struct {
	sync.RWMutex
	store map[string]CacheItem
}

var cacheEngine = &MemoryCache{store: make(map[string]CacheItem)}

func (mc *MemoryCache) Get(key string) ([]byte, bool) {
	mc.RLock()
	defer mc.RUnlock()
	item, found := mc.store[key]
	if !found {
		return nil, false
	}
	if time.Since(item.Timestamp).Seconds() < CacheTTL {
		return item.Data, true
	}
	return nil, false
}

func (mc *MemoryCache) Set(key string, data []byte) {
	mc.Lock()
	defer mc.Unlock()
	mc.store[key] = CacheItem{Timestamp: time.Now(), Data: data}
}

func (mc *MemoryCache) Clean() {
	mc.Lock()
	defer mc.Unlock()
	now := time.Now()
	for k, v := range mc.store {
		if now.Sub(v.Timestamp).Seconds() >= CacheTTL {
			delete(mc.store, k)
		}
	}
}

func (mc *MemoryCache) ClearAll() {
	mc.Lock()
	defer mc.Unlock()
	mc.store = make(map[string]CacheItem)
}

func executeYtDlp(query string, cookie string, attempt int) (map[string]interface{}, string, error) {
	args := []string{
		"--dump-json",
		"--no-warnings",
		"--skip-download",
		"--no-playlist",
		"--socket-timeout", "15",
		"--compat-options", "no-youtube-unavailable-videos",
		"--geo-bypass",
		"--impersonate", "chrome",
	}

	if cookie != "" {
		args = append(args, "--cookies", cookie)
	}

	method := ""
	if attempt == 1 {
		args = append(args, "--extractor-args", "youtube:player_client=web", "--remote-components", "ejs:github,ejs:npm")
		method = "Go Engine (Web Client + Chrome Spoofing)"
	} else if attempt == 2 {
		args = append(args, "--extractor-args", "youtube:player_client=web,android;player_skip=configs", "--remote-components", "ejs:github,ejs:npm")
		method = "Go Engine (Web/Android + Chrome Spoofing)"
	} else {
		args = append(args, "--extractor-args", "youtube:player_client=web")
		method = "Go Engine (Web Fallback + Chrome Spoofing)"
	}

	if !strings.HasPrefix(query, "http") && !strings.HasPrefix(query, "ytsearch") {
		args = append(args, "ytsearch1:"+query)
	} else {
		args = append(args, query)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, YtDlpBin, args...)
	
	var out bytes.Buffer
	var stderr bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &stderr

	err := cmd.Run()
	if err != nil {
		errMsg := strings.TrimSpace(stderr.String())
		if errMsg == "" {
			errMsg = err.Error()
		}
		return nil, "", fmt.Errorf("YT-DLP Error: %s", errMsg)
	}

	var result map[string]interface{}
	err = json.Unmarshal(out.Bytes(), &result)
	if err != nil {
		return nil, "", fmt.Errorf("JSON Parse Error: %v", err)
	}

	if entries, ok := result["entries"].([]interface{}); ok && len(entries) > 0 {
		if first, ok := entries[0].(map[string]interface{}); ok {
			return first, method, nil
		}
	}

	return result, method, nil
}

func extractSmart(query string) (map[string]interface{}, string, error) {
	var lastErr error
	for attempt := 1; attempt <= 3; attempt++ {
		cookie := cookieVault.GetCookie()
		info, method, err := executeYtDlp(query, cookie, attempt)
		if err == nil && info != nil {
			if _, ok := info["formats"]; ok {
				return info, method, nil
			}
		}
		lastErr = err
		if cookie != "" && err != nil && (strings.Contains(strings.ToLower(err.Error()), "sign in") || strings.Contains(strings.ToLower(err.Error()), "bot")) {
			cookieVault.ReportFailure(cookie)
		}
		time.Sleep(300 * time.Millisecond)
	}
	return nil, "", lastErr
}

func buildSmartFormats(rawFormats []interface{}) (SmartFormats, string, string) {
	sf := SmartFormats{
		BestMuxed: make([]StreamInfo, 0, 10),
		AudioOnly: make([]StreamInfo, 0, 10),
		VideoOnly: make([]StreamInfo, 0, 10),
	}

	for _, rf := range rawFormats {
		f, ok := rf.(map[string]interface{})
		if !ok {
			continue
		}

		ext, _ := f["ext"].(string)
		proto, _ := f["protocol"].(string)
		if ext == "mhtml" || ext == "sb0" || ext == "sb1" || strings.HasPrefix(proto, "mhtml") {
			continue
		}
		if !strings.HasPrefix(proto, "http") && !strings.HasPrefix(proto, "m3u8") {
			continue
		}

		vcodec, _ := f["vcodec"].(string)
		acodec, _ := f["acodec"].(string)
		if vcodec == "" {
			vcodec = "none"
		}
		if acodec == "" {
			acodec = "none"
		}

		url, _ := f["url"].(string)
		if url == "" {
			continue
		}

		fmtID, _ := f["format_id"].(string)
		res, _ := f["format_note"].(string)
		if res == "" {
			res, _ = f["resolution"].(string)
		}

		score := 0
		if strings.Contains(ext, "mp4") || strings.Contains(ext, "m4a") {
			score += 10
		}
		if tbr, ok := f["tbr"].(float64); ok {
			score += int(tbr / 100)
		}

		info := StreamInfo{
			FormatID:     fmtID,
			Ext:          ext,
			Resolution:   res,
			Vcodec:       vcodec,
			Acodec:       acodec,
			URL:          url,
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

	bestA, bestV := "", ""
	if len(sf.AudioOnly) > 0 {
		bestA = sf.AudioOnly[0].URL
	}
	if len(sf.BestMuxed) > 0 {
		bestV = sf.BestMuxed[0].URL
	} else if len(sf.VideoOnly) > 0 {
		bestV = sf.VideoOnly[0].URL
	}

	return sf, bestA, bestV
}

func verifyAuth(r *http.Request) bool {
	key := r.Header.Get("X-Titan-Key")
	if key == "" {
		key = r.Header.Get("X-Ultra-Key")
	}
	return key == APIKey
}

func enableCors(w *http.ResponseWriter) {
	(*w).Header().Set("Access-Control-Allow-Origin", "*")
	(*w).Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
	(*w).Header().Set("Access-Control-Allow-Headers", "*")
}

func handleExtract(w http.ResponseWriter, r *http.Request) {
	enableCors(&w)
	if r.Method == "OPTIONS" {
		return
	}

	atomic.AddUint64(&Stats.TotalRequests, 1)
	start := time.Now()

	if !verifyAuth(r) {
		atomic.AddUint64(&Stats.FailedRequests, 1)
		http.Error(w, `{"success":false,"message":"Forbidden"}`, http.StatusForbidden)
		return
	}

	url := r.URL.Query().Get("url")
	audioOnlyStr := r.URL.Query().Get("audio_only")
	forceRefreshStr := r.URL.Query().Get("force_refresh")

	audioOnly := audioOnlyStr != "false" && audioOnlyStr != "0"
	forceRefresh := forceRefreshStr == "true" || forceRefreshStr == "1"

	if url == "" {
		atomic.AddUint64(&Stats.FailedRequests, 1)
		http.Error(w, `{"success":false,"message":"Missing URL"}`, http.StatusBadRequest)
		return
	}

	cacheKey := fmt.Sprintf("%s_audio:%v", url, audioOnly)

	if !forceRefresh {
		if cachedData, found := cacheEngine.Get(cacheKey); found {
			var resp MediaResponse
			json.Unmarshal(cachedData, &resp)
			resp.Cached = true
			resp.ProcessTime = float64(time.Since(start).Milliseconds())
			atomic.AddUint64(&Stats.SuccessfulRequests, 1)
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(resp)
			return
		}
	}

	info, method, err := extractSmart(url)
	if err != nil {
		atomic.AddUint64(&Stats.FailedRequests, 1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]interface{}{
			"success":         false,
			"message":         err.Error(),
			"process_time_ms": float64(time.Since(start).Milliseconds()),
		})
		return
	}

	rawFormats, _ := info["formats"].([]interface{})
	sf, bestA, bestV := buildSmartFormats(rawFormats)

	directURL := ""
	if audioOnly {
		directURL = bestA
		if directURL == "" {
			directURL = bestV
		}
	} else {
		directURL = bestV
		if directURL == "" {
			directURL = bestA
		}
	}

	if directURL == "" {
		directURL, _ = info["url"].(string)
	}

	isLive := false
	if live, ok := info["is_live"].(bool); ok {
		isLive = live
	} else if wasLive, ok := info["was_live"].(bool); ok {
		isLive = wasLive
	}

	duration := 0
	if dur, ok := info["duration"].(float64); ok && !isLive {
		duration = int(dur)
	}

	vidID, _ := info["id"].(string)
	title, _ := info["title"].(string)

	thumbs := make([]ThumbnailModel, 0)
	if rawThumbs, ok := info["thumbnails"].([]interface{}); ok {
		for _, rt := range rawThumbs {
			if tmap, ok := rt.(map[string]interface{}); ok {
				u, _ := tmap["url"].(string)
				w, _ := tmap["width"].(float64)
				h, _ := tmap["height"].(float64)
				thumbs = append(thumbs, ThumbnailModel{URL: u, Width: int(w), Height: int(h)})
			}
		}
	}

	resp := MediaResponse{
		Success:          true,
		ProcessTime:      float64(time.Since(start).Milliseconds()),
		Cached:           false,
		ExtractionMethod: method,
		VideoID:          vidID,
		Title:            title,
		Duration:         duration,
		IsLive:           isLive,
		Thumbnails:       thumbs,
		DirectStreamURL:  directURL,
		SmartFormats:     sf,
		RawFallbackCount: len(rawFormats),
	}

	respBytes, _ := json.Marshal(resp)
	cacheEngine.Set(cacheKey, respBytes)

	atomic.AddUint64(&Stats.SuccessfulRequests, 1)
	w.Header().Set("Content-Type", "application/json")
	w.Write(respBytes)
}

func handleWS(w http.ResponseWriter, r *http.Request) {
	DynamicConfig.RLock()
	allow := DynamicConfig.AllowNewWS
	DynamicConfig.RUnlock()

	if !allow {
		http.Error(w, "Connections disabled", http.StatusServiceUnavailable)
		return
	}

	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		return
	}
	defer conn.Close()

	atomic.AddInt64(&Stats.ActiveWS, 1)
	defer atomic.AddInt64(&Stats.ActiveWS, -1)

	var authMsg map[string]string
	if err := conn.ReadJSON(&authMsg); err != nil || authMsg["auth"] != APIKey {
		conn.Close()
		return
	}

	for {
		var req map[string]interface{}
		if err := conn.ReadJSON(&req); err != nil {
			break
		}

		url, _ := req["url"].(string)
		if url == "" {
			conn.WriteJSON(map[string]interface{}{"success": false, "message": "Missing URL"})
			continue
		}

		audioOnly := true
		if ao, ok := req["audio_only"].(bool); ok {
			audioOnly = ao
		}

		atomic.AddUint64(&Stats.TotalRequests, 1)
		start := time.Now()
		cacheKey := fmt.Sprintf("%s_audio:%v", url, audioOnly)

		if cachedData, found := cacheEngine.Get(cacheKey); found {
			var resp MediaResponse
			json.Unmarshal(cachedData, &resp)
			resp.Cached = true
			resp.ProcessTime = float64(time.Since(start).Milliseconds())
			atomic.AddUint64(&Stats.SuccessfulRequests, 1)
			conn.WriteJSON(resp)
			continue
		}

		info, method, err := extractSmart(url)
		if err != nil {
			atomic.AddUint64(&Stats.FailedRequests, 1)
			conn.WriteJSON(map[string]interface{}{
				"success":         false,
				"message":         err.Error(),
				"process_time_ms": float64(time.Since(start).Milliseconds()),
			})
			continue
		}

		rawFormats, _ := info["formats"].([]interface{})
		sf, bestA, bestV := buildSmartFormats(rawFormats)

		directURL := bestA
		if !audioOnly {
			directURL = bestV
		}
		if directURL == "" {
			directURL, _ = info["url"].(string)
		}

		isLive := false
		if live, ok := info["is_live"].(bool); ok {
			isLive = live
		} else if wasLive, ok := info["was_live"].(bool); ok {
			isLive = wasLive
		}

		duration := 0
		if dur, ok := info["duration"].(float64); ok && !isLive {
			duration = int(dur)
		}
		vidID, _ := info["id"].(string)
		title, _ := info["title"].(string)

		resp := MediaResponse{
			Success:          true,
			ProcessTime:      float64(time.Since(start).Milliseconds()),
			Cached:           false,
			ExtractionMethod: method,
			VideoID:          vidID,
			Title:            title,
			Duration:         duration,
			IsLive:           isLive,
			DirectStreamURL:  directURL,
			SmartFormats:     sf,
		}

		respBytes, _ := json.Marshal(resp)
		cacheEngine.Set(cacheKey, respBytes)
		atomic.AddUint64(&Stats.SuccessfulRequests, 1)
		conn.WriteJSON(resp)
	}
}

func handleAdminConfig(w http.ResponseWriter, r *http.Request) {
	enableCors(&w)
	if r.Method == "OPTIONS" {
		return
	}
	if !verifyAuth(r) || r.Method != "POST" {
		http.Error(w, `{"success":false,"message":"Forbidden"}`, http.StatusForbidden)
		return
	}
	var req map[string]interface{}
	if err := json.NewDecoder(r.Body).Decode(&req); err == nil {
		DynamicConfig.Lock()
		if val, ok := req["smart_validation_enabled"].(bool); ok {
			DynamicConfig.SmartValidation = val
		}
		if val, ok := req["allow_new_ws_connections"].(bool); ok {
			DynamicConfig.AllowNewWS = val
		}
		DynamicConfig.Unlock()
	}
	w.Header().Set("Content-Type", "application/json")
	DynamicConfig.RLock()
	json.NewEncoder(w).Encode(map[string]interface{}{
		"success":        true,
		"current_config": DynamicConfig,
	})
	DynamicConfig.RUnlock()
}

func main() {
	go func() {
		for {
			time.Sleep(2 * time.Minute)
			cacheEngine.Clean()
			cookieVault.Refresh()
		}
	}()

	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		enableCors(&w)
		if r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}
		if _, err := os.Stat("index.html"); err == nil {
			http.ServeFile(w, r, "index.html")
		} else {
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{"system": "Go Fast Engine", "status": "Active"}`))
		}
	})

	http.HandleFunc("/api/v1/extract", handleExtract)
	http.HandleFunc("/api/v1/ws/stream", handleWS)
	http.HandleFunc("/api/v1/admin/config", handleAdminConfig)

	http.HandleFunc("/api/v1/admin/stats", func(w http.ResponseWriter, r *http.Request) {
		enableCors(&w)
		if !verifyAuth(r) {
			http.Error(w, "Forbidden", http.StatusForbidden)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"status":                  "online",
			"uptime_seconds":          time.Since(ServerStart).Seconds(),
			"total_requests":          atomic.LoadUint64(&Stats.TotalRequests),
			"successful_requests":     atomic.LoadUint64(&Stats.SuccessfulRequests),
			"failed_requests":         atomic.LoadUint64(&Stats.FailedRequests),
			"active_ws_connections":   atomic.LoadInt64(&Stats.ActiveWS),
			"cached_files_count":      len(cacheEngine.store),
			"memory_usage":            getMemoryUsage(),
			"available_cookies":       len(cookieVault.pool),
		})
	})

	http.HandleFunc("/api/v1/admin/clear_cache", func(w http.ResponseWriter, r *http.Request) {
		enableCors(&w)
		if !verifyAuth(r) || r.Method != "POST" {
			http.Error(w, "Forbidden", http.StatusForbidden)
			return
		}
		cacheEngine.ClearAll()
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"success": true, "message": "Cache completely cleared."}`))
	})

	port := getEnv("PORT", "8080")
	http.ListenAndServe("0.0.0.0:"+port, nil)
}

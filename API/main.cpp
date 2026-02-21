#include <drogon/drogon.h>
#include <drogon/WebSocketController.h>
#include <nlohmann/json.hpp>
#include <chrono>
#include <fstream>
#include <sstream>
#include <thread>
#include <mutex>
#include <cstdlib>
#include "models.hpp"
#include "cookie_manager.hpp"
#include "smart_cache.hpp"
#include "core_engine.hpp"

using namespace drogon;
using json = nlohmann::json;

EnterpriseCookieManager cookie_vault;
SmartCache cache_engine;
CoreEngine core_engine;

std::string API_KEY = "Ultra_2026_Fast";
bool SMART_VALIDATION = true;
int VALIDATION_INTERVAL = 7200;
bool ALLOW_WS = true;

struct Stats {
    std::atomic<uint64_t> total_requests{0};
    std::atomic<uint64_t> successful_requests{0};
    std::atomic<uint64_t> failed_requests{0};
    std::atomic<uint64_t> active_ws{0};
    std::chrono::time_point<std::chrono::system_clock> start_time = std::chrono::system_clock::now();
} api_stats;

std::string get_memory_usage() {
    std::ifstream file("/proc/meminfo");
    std::string line;
    long total = 0, available = 0;
    while (std::getline(file, line)) {
        if (line.find("MemTotal:") == 0) {
            std::sscanf(line.c_str(), "MemTotal: %ld kB", &total);
        } else if (line.find("MemAvailable:") == 0) {
            std::sscanf(line.c_str(), "MemAvailable: %ld kB", &available);
        }
    }
    if (total > 0) {
        long used = total - available;
        return std::to_string(used / 1024) + " MB / " + std::to_string(total / 1024) + " MB";
    }
    return "Unknown";
}

bool verify_auth(const HttpRequestPtr &req) {
    auto key = req->getHeader("X-Ultra-Key");
    return key == API_KEY;
}

void add_cors(const HttpResponsePtr &resp) {
    resp->addHeader("Access-Control-Allow-Origin", "*");
    resp->addHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    resp->addHeader("Access-Control-Allow-Headers", "X-Ultra-Key, Content-Type");
}

void smart_validator_task() {
    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(VALIDATION_INTERVAL));
        if (!SMART_VALIDATION) continue;
        auto all_items = cache_engine.get_all_items();
        for (const auto& [key, item] : all_items) {
            try {
                std::string url = item.data.value("direct_stream_url", "");
                if (!url.empty()) {
                    auto client = HttpClient::newHttpClient(url);
                    auto req = HttpRequest::newHttpRequest();
                    req->setMethod(Head);
                    client->sendRequest(req, [key](ReqResult res, const HttpResponsePtr& resp) {
                        if (res == ReqResult::Ok && resp) {
                            if (resp->getStatusCode() == k403Forbidden || resp->getStatusCode() == k404NotFound || resp->getStatusCode() == k410Gone) {
                                cache_engine.remove_key(key);
                            }
                        } else {
                            cache_engine.remove_key(key);
                        }
                    });
                }
            } catch (...) {}
        }
    }
}

class UltraWSController : public WebSocketController<UltraWSController> {
public:
    void handleNewMessage(const WebSocketConnectionPtr& wsConn, std::string&& message, const WebSocketMessageType& type) override {
        if (type != WebSocketMessageType::Text) return;
        try {
            api_stats.total_requests++;
            auto start = std::chrono::high_resolution_clock::now();
            json req = json::parse(message);
            if (req.contains("auth") && req["auth"] == API_KEY) {
                wsConn->setContext(std::make_shared<bool>(true));
                return;
            }
            auto ctx = wsConn->getContext<bool>();
            if (!ctx || !(*ctx)) {
                wsConn->forceClose();
                return;
            }
            std::string url = req.value("url", "");
            bool audio_only = req.value("audio_only", true);
            if (url.empty()) return;

            std::string cache_key = url + "_audio:" + (audio_only ? "1" : "0");
            auto cached = cache_engine.get(cache_key);
            if (cached.has_value()) {
                api_stats.successful_requests++;
                auto end = std::chrono::high_resolution_clock::now();
                json resp = cached.value();
                resp["cached"] = true;
                resp["process_time_ms"] = std::chrono::duration<double, std::milli>(end - start).count();
                wsConn->send(resp.dump());
                return;
            }
            
            std::string cookie = cookie_vault.get_cookie();
            json raw_info = core_engine.extract_raw(url, cookie, 1);
            std::string best_audio, best_video;
            SmartFormats sf = core_engine.build_smart_formats(raw_info.value("formats", json::array()), best_audio, best_video);
            
            std::string direct_url = audio_only ? (best_audio.empty() ? best_video : best_audio) : (best_video.empty() ? best_audio : best_video);
            if (direct_url.empty()) direct_url = raw_info.value("url", "");

            json response;
            response["success"] = true;
            response["video_id"] = raw_info.value("id", "");
            response["title"] = raw_info.value("title", "Unknown");
            response["direct_stream_url"] = direct_url;
            
            auto end = std::chrono::high_resolution_clock::now();
            response["process_time_ms"] = std::chrono::duration<double, std::milli>(end - start).count();
            
            cache_engine.set(cache_key, response);
            api_stats.successful_requests++;
            wsConn->send(response.dump());
        } catch (...) {
            api_stats.failed_requests++;
        }
    }
    void handleNewConnection(const HttpRequestPtr& req, const WebSocketConnectionPtr& wsConn) override {
        if (!ALLOW_WS) {
            wsConn->forceClose();
            return;
        }
        api_stats.active_ws++;
    }
    void handleConnectionClosed(const WebSocketConnectionPtr& wsConn) override {
        api_stats.active_ws--;
    }
    WS_PATH_LIST_BEGIN
    WS_PATH_ADD("/api/v1/ws/stream");
    WS_PATH_LIST_END
};

int main() {
    if (const char* env_key = std::getenv("ULTRA_SECRET_KEY")) API_KEY = env_key;
    
    std::thread(smart_validator_task).detach();

    app().registerHandler("/", [](const HttpRequestPtr& req, std::function<void(const HttpResponsePtr&)>&& callback) {
        std::ifstream file("index.html");
        if (file.is_open()) {
            std::stringstream buffer;
            buffer << file.rdbuf();
            auto resp = HttpResponse::newHttpResponse();
            resp->setBody(buffer.str());
            resp->setContentTypeCode(CT_TEXT_HTML);
            add_cors(resp);
            callback(resp);
        } else {
            auto resp = HttpResponse::newHttpResponse();
            json j = {{"system", "UltraCore"}, {"status", "Active"}, {"engine", "Zero Latency Engine"}};
            resp->setBody(j.dump());
            resp->setContentTypeCode(CT_APPLICATION_JSON);
            add_cors(resp);
            callback(resp);
        }
    });

    app().registerHandler("/api/v1/admin/stats", [](const HttpRequestPtr& req, std::function<void(const HttpResponsePtr&)>&& callback) {
        if (!verify_auth(req)) {
            auto resp = HttpResponse::newHttpResponse();
            resp->setStatusCode(k403Forbidden);
            callback(resp);
            return;
        }
        auto now = std::chrono::system_clock::now();
        double uptime = std::chrono::duration<double>(now - api_stats.start_time).count();
        json j = {
            {"status", "online"},
            {"uptime_seconds", uptime},
            {"total_requests", api_stats.total_requests.load()},
            {"successful_requests", api_stats.successful_requests.load()},
            {"failed_requests", api_stats.failed_requests.load()},
            {"active_ws_connections", api_stats.active_ws.load()},
            {"cached_files_count", cache_engine.size()},
            {"memory_usage", get_memory_usage()},
            {"available_cookies", cookie_vault.get_pool_size()}
        };
        auto resp = HttpResponse::newHttpResponse();
        resp->setBody(j.dump());
        resp->setContentTypeCode(CT_APPLICATION_JSON);
        add_cors(resp);
        callback(resp);
    });

    app().registerHandler("/api/v1/admin/clear_cache", [](const HttpRequestPtr& req, std::function<void(const HttpResponsePtr&)>&& callback) {
        if (!verify_auth(req)) {
            auto resp = HttpResponse::newHttpResponse();
            resp->setStatusCode(k403Forbidden);
            callback(resp);
            return;
        }
        cache_engine.clear_all();
        json j = {{"success", true}};
        auto resp = HttpResponse::newHttpResponse();
        resp->setBody(j.dump());
        resp->setContentTypeCode(CT_APPLICATION_JSON);
        add_cors(resp);
        callback(resp);
    }, {Post});

    app().registerHandler("/api/v1/extract", [](const HttpRequestPtr& req, std::function<void(const HttpResponsePtr&)>&& callback) {
        api_stats.total_requests++;
        auto start = std::chrono::high_resolution_clock::now();
        
        if (!verify_auth(req)) {
            api_stats.failed_requests++;
            auto resp = HttpResponse::newHttpResponse();
            resp->setStatusCode(k403Forbidden);
            callback(resp);
            return;
        }

        std::string url = req->getParameter("url");
        std::string audio_param = req->getParameter("audio_only");
        std::string refresh_param = req->getParameter("force_refresh");
        bool audio_only = (audio_param.empty() || audio_param == "true" || audio_param == "1");
        bool force_refresh = (refresh_param == "true" || refresh_param == "1");

        if (url.empty()) {
            api_stats.failed_requests++;
            json err = {{"success", false}, {"message", "Missing URL"}};
            auto resp = HttpResponse::newHttpResponse();
            resp->setBody(err.dump());
            resp->setContentTypeCode(CT_APPLICATION_JSON);
            resp->setStatusCode(k400BadRequest);
            add_cors(resp);
            callback(resp);
            return;
        }

        std::string cache_key = url + "_audio:" + (audio_only ? "1" : "0");

        if (!force_refresh) {
            auto cached = cache_engine.get(cache_key);
            if (cached.has_value()) {
                api_stats.successful_requests++;
                json c = cached.value();
                c["cached"] = true;
                auto end = std::chrono::high_resolution_clock::now();
                c["process_time_ms"] = std::chrono::duration<double, std::milli>(end - start).count();
                auto resp = HttpResponse::newHttpResponse();
                resp->setBody(c.dump());
                resp->setContentTypeCode(CT_APPLICATION_JSON);
                add_cors(resp);
                callback(resp);
                return;
            }
        }

        app().getLoop()->queueInLoop([url, audio_only, cache_key, start, callback]() {
            try {
                std::string cookie = cookie_vault.get_cookie();
                json raw_info = core_engine.extract_raw(url, cookie, 1);
                
                std::string best_audio, best_video;
                SmartFormats sf = core_engine.build_smart_formats(raw_info.value("formats", json::array()), best_audio, best_video);
                
                std::string direct_url = audio_only ? (best_audio.empty() ? best_video : best_audio) : (best_video.empty() ? best_audio : best_video);
                if (direct_url.empty()) direct_url = raw_info.value("url", "");
                if (direct_url.empty()) throw std::runtime_error("No valid streams found");

                MediaResponse response_obj;
                response_obj.success = true;
                response_obj.cached = false;
                response_obj.extraction_method = "Ultra Engine";
                response_obj.video_id = raw_info.value("id", "");
                response_obj.title = raw_info.value("title", "Unknown");
                response_obj.duration = raw_info.value("duration", 0);
                response_obj.is_live = raw_info.value("is_live", false);
                response_obj.direct_stream_url = direct_url;
                response_obj.smart_formats = sf;

                json response_json = response_obj;
                auto end = std::chrono::high_resolution_clock::now();
                response_json["process_time_ms"] = std::chrono::duration<double, std::milli>(end - start).count();

                cache_engine.set(cache_key, response_json);
                api_stats.successful_requests++;

                auto resp = HttpResponse::newHttpResponse();
                resp->setBody(response_json.dump());
                resp->setContentTypeCode(CT_APPLICATION_JSON);
                add_cors(resp);
                callback(resp);
            } catch (const std::exception& e) {
                api_stats.failed_requests++;
                json err = {{"success", false}, {"message", e.what()}};
                auto resp = HttpResponse::newHttpResponse();
                resp->setBody(err.dump());
                resp->setContentTypeCode(CT_APPLICATION_JSON);
                resp->setStatusCode(k500InternalServerError);
                add_cors(resp);
                callback(resp);
            }
        });
    });

    app().addListener("0.0.0.0", 8080);
    app().setThreadNum(10);
    app().run();
    return 0;
}

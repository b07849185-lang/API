#pragma once
#include <string>
#include <vector>
#include <stdexcept>
#include <array>
#include <memory>
#include <algorithm>
#include <iostream>
#include <nlohmann/json.hpp>
#include "models.hpp"

using json = nlohmann::json;

class CoreEngine {
private:
    std::string execute_cmd(const std::string& cmd) {
        std::array<char, 2048> buffer;
        std::string result;
        std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
        if (!pipe) {
            throw std::runtime_error("popen failed");
        }
        while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
            result += buffer.data();
        }
        return result;
    }

public:
    CoreEngine() {}

    json extract_raw(const std::string& raw_query, const std::string& cookie_path, int attempt) {
        std::string cmd = "yt-dlp --dump-json --no-warnings --skip-download --no-playlist --default-search ytsearch --socket-timeout 8 --compat-options no-youtube-unavailable-videos ";
        
        if (!cookie_path.empty()) {
            cmd += "--cookies \"" + cookie_path + "\" ";
        }

        if (attempt == 1 || attempt == 3) {
            cmd += "--extractor-args \"youtube:player_client=web\" --remote-components ejs:github,ejs:npm ";
        } else if (attempt == 2) {
            cmd += "--extractor-args \"youtube:player_client=web,android;player_skip=configs\" --remote-components ejs:github,ejs:npm ";
        }

        std::string final_query = raw_query;
        if (final_query.find("http") != 0 && final_query.find("ytsearch") != 0) {
            final_query = "ytsearch1:\"" + final_query + "\"";
        } else {
            final_query = "\"" + final_query + "\"";
        }
        cmd += final_query;

        std::string output = execute_cmd(cmd);
        
        if (output.empty()) {
            throw std::runtime_error("Extraction failed: Engine returned empty response");
        }

        try {
            json j = json::parse(output);
            if (j.contains("entries") && j["entries"].is_array() && !j["entries"].empty()) {
                return j["entries"][0];
            }
            return j;
        } catch (const json::parse_error& e) {
            throw std::runtime_error("JSON Parse Error");
        }
    }

    SmartFormats build_smart_formats(const json& formats_list, std::string& best_audio, std::string& best_video) {
        SmartFormats sf;
        if (!formats_list.is_array()) return sf;

        for (const auto& f : formats_list) {
            try {
                std::string ext = f.value("ext", "");
                if (ext == "mhtml" || ext == "sb0" || ext == "sb1") continue;

                std::string proto = f.value("protocol", "");
                if (proto.find("http") != 0 && proto.find("m3u8") != 0) continue;

                std::string vcodec = f.value("vcodec", "none");
                std::string acodec = f.value("acodec", "none");
                std::string url = f.value("url", "");
                if (url.empty()) continue;

                StreamInfo info;
                info.format_id = f.value("format_id", "0");
                info.ext = ext;
                info.resolution = f.contains("format_note") ? f["format_note"].get<std::string>() : f.value("resolution", "unknown");
                info.vcodec = vcodec;
                info.acodec = acodec;
                info.url = url;

                int score = 0;
                if (ext.find("mp4") != std::string::npos || ext.find("m4a") != std::string::npos) score += 10;
                double tbr = f.value("tbr", 0.0);
                score += static_cast<int>(tbr / 100);

                if (vcodec != "none" && acodec != "none") {
                    if (proto.find("m3u8") == std::string::npos) score += 50;
                    info.quality_score = score;
                    sf.best_muxed.push_back(info);
                } else if (vcodec == "none" && acodec != "none") {
                    info.quality_score = score;
                    sf.audio_only.push_back(info);
                } else if (vcodec != "none" && acodec == "none") {
                    info.quality_score = score;
                    sf.video_only.push_back(info);
                }
            } catch (...) {
                continue;
            }
        }

        auto sort_desc = [](const StreamInfo& a, const StreamInfo& b) { return a.quality_score > b.quality_score; };
        std::sort(sf.best_muxed.begin(), sf.best_muxed.end(), sort_desc);
        std::sort(sf.audio_only.begin(), sf.audio_only.end(), sort_desc);
        std::sort(sf.video_only.begin(), sf.video_only.end(), sort_desc);

        best_audio = sf.audio_only.empty() ? "" : sf.audio_only[0].url;
        best_video = sf.best_muxed.empty() ? (sf.video_only.empty() ? "" : sf.video_only[0].url) : sf.best_muxed[0].url;

        return sf;
    }
};

extern CoreEngine core_engine;

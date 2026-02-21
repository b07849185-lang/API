#pragma once
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

struct StreamInfo {
    std::string format_id;
    std::string ext;
    std::string resolution;
    std::string vcodec;
    std::string acodec;
    std::string url;
    int quality_score = 0;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(StreamInfo, format_id, ext, resolution, vcodec, acodec, url, quality_score)
};

struct SmartFormats {
    std::vector<StreamInfo> best_muxed;
    std::vector<StreamInfo> audio_only;
    std::vector<StreamInfo> video_only;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(SmartFormats, best_muxed, audio_only, video_only)
};

struct ThumbnailModel {
    std::string url;
    int width = 0;
    int height = 0;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(ThumbnailModel, url, width, height)
};

struct MediaResponse {
    bool success = true;
    double process_time_ms = 0.0;
    bool cached = false;
    std::string extraction_method;
    std::string video_id;
    std::string title;
    int duration = 0;
    bool is_live = false;
    std::vector<ThumbnailModel> thumbnails;
    std::string direct_stream_url;
    SmartFormats smart_formats;
    int raw_fallback_count = 0;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(MediaResponse, success, process_time_ms, cached, extraction_method, video_id, title, duration, is_live, thumbnails, direct_stream_url, smart_formats, raw_fallback_count)
};

struct ErrorResponse {
    bool success = false;
    int error_code = 500;
    std::string message;
    double process_time_ms = 0.0;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(ErrorResponse, success, error_code, message, process_time_ms)
};

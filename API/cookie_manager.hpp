#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <filesystem>
#include <random>
#include <chrono>
#include <algorithm>

namespace fs = std::filesystem;

class EnterpriseCookieManager {
private:
    std::string directory;
    std::vector<std::string> pool;
    std::unordered_map<std::string, double> banned;
    std::unordered_map<std::string, double> last_used;
    int ban_time;
    int reuse_delay;
    std::mutex mtx; 

    double get_current_time() {
        auto now = std::chrono::system_clock::now().time_since_epoch();
        return std::chrono::duration_cast<std::chrono::seconds>(now).count();
    }

    void _refresh_pool() {
        double now = get_current_time();
        
        for (auto it = banned.begin(); it != banned.end(); ) {
            if (now - it->second > ban_time) {
                it = banned.erase(it);
            } else {
                ++it;
            }
        }

        pool.clear();
        if (fs::exists(directory)) {
            for (const auto& entry : fs::directory_iterator(directory)) {
                if (entry.path().extension() == ".txt" && entry.file_size() > 0) {
                    std::string file_path = entry.path().string();
                    if (banned.find(file_path) == banned.end()) {
                        pool.push_back(file_path);
                    }
                }
            }
        }
    }

public:
    EnterpriseCookieManager(std::string dir = "cookies", int ban = 3600, int reuse = 2) 
        : directory(dir), ban_time(ban), reuse_delay(reuse) {
        if (!fs::exists(directory)) {
            fs::create_directory(directory);
        }
        _refresh_pool();
    }

    void refresh_pool_sync() {
        std::lock_guard<std::mutex> lock(mtx);
        _refresh_pool();
    }

    std::string get_cookie() {
        std::lock_guard<std::mutex> lock(mtx);
        
        if (pool.empty()) {
            _refresh_pool(); 
            if (pool.empty()) return "";
        }

        double now = get_current_time();
        std::vector<std::string> candidates;
        
        for (const auto& c : pool) {
            if (now - last_used[c] > reuse_delay) {
                candidates.push_back(c);
            }
        }

        if (candidates.empty()) candidates = pool;
        if (candidates.empty()) return "";

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> distr(0, candidates.size() - 1);
        
        std::string chosen = candidates[distr(gen)];
        last_used[chosen] = now;
        return chosen;
    }

    void report_failure(const std::string& cookie_path) {
        std::lock_guard<std::mutex> lock(mtx);
        banned[cookie_path] = get_current_time();
        pool.erase(std::remove(pool.begin(), pool.end(), cookie_path), pool.end());
    }
    
    int get_pool_size() {
        std::lock_guard<std::mutex> lock(mtx);
        return pool.size();
    }
};

extern EnterpriseCookieManager cookie_vault;

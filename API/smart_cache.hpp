#pragma once
#include <unordered_map>
#include <string>
#include <mutex>
#include <shared_mutex>
#include <chrono>
#include <optional>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// هيكل يحفظ البيانات ووقت حفظها
struct CacheItem {
    double timestamp;
    json data;
};

class SmartCache {
private:
    std::unordered_map<std::string, CacheItem> cache_store;
    // shared_mutex بيسمح بقراءات متعددة في نفس الوقت، بس بيكتب حاجة واحدة بس
    mutable std::shared_mutex rw_mtx; 
    int cache_ttl;

    // دالة لحساب الوقت الحالي بالثواني
    double get_current_time() const {
        auto now = std::chrono::system_clock::now().time_since_epoch();
        return std::chrono::duration_cast<std::chrono::seconds>(now).count();
    }

public:
    // الإعداد الافتراضي لمدة بقاء الكاش (14400 ثانية = 4 ساعات)
    SmartCache(int ttl = 14400) : cache_ttl(ttl) {}

    // استرجاع البيانات (قراءة سريعة جداً ومفتوحة للكل)
    std::optional<json> get(const std::string& key) {
        std::shared_lock<std::shared_mutex> lock(rw_mtx); // قفل قراءة فقط
        auto it = cache_store.find(key);
        if (it != cache_store.end()) {
            if (get_current_time() - it->second.timestamp < cache_ttl) {
                return it->second.data; // الرابط لسه شغال
            }
        }
        return std::nullopt; // مفيش داتا أو منتهية الصلاحية
    }

    // حفظ بيانات جديدة (قفل حصري عشان نمنع التداخل)
    void set(const std::string& key, const json& data) {
        std::unique_lock<std::shared_mutex> lock(rw_mtx); // قفل كتابة حصري
        cache_store[key] = {get_current_time(), data};
    }

    // تنظيف الكاش من الروابط الميتة (بيشتغل في الخلفية)
    void cleanup() {
        std::unique_lock<std::shared_mutex> lock(rw_mtx);
        double now = get_current_time();
        for (auto it = cache_store.begin(); it != cache_store.end(); ) {
            if (now - it->second.timestamp >= cache_ttl) {
                it = cache_store.erase(it);
            } else {
                ++it;
            }
        }
    }

    // مسح الكاش بالكامل (عشان زرار لوحة التحكم)
    void clear_all() {
        std::unique_lock<std::shared_mutex> lock(rw_mtx);
        cache_store.clear();
    }

    // مسح رابط معين (بيستخدمه منظف الكاش الذكي)
    void remove_key(const std::string& key) {
        std::unique_lock<std::shared_mutex> lock(rw_mtx);
        cache_store.erase(key);
    }

    // جلب كل الكاش (للوحة التحكم والمراقبة)
    std::unordered_map<std::string, CacheItem> get_all_items() {
        std::shared_lock<std::shared_mutex> lock(rw_mtx);
        return cache_store;
    }
    
    // إرجاع عدد الروابط المحفوظة
    int size() {
        std::shared_lock<std::shared_mutex> lock(rw_mtx);
        return cache_store.size();
    }
};

// إنشاء النسخة الموحدة (Global Instance)
extern SmartCache cache_engine;

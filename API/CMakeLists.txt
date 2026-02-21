cmake_minimum_required(VERSION 3.15)
project(UltraCore_Engine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Drogon CONFIG REQUIRED)

include(FetchContent)
FetchContent_Declare(json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(json)

add_executable(UltraServer main.cpp)

target_link_libraries(UltraServer PRIVATE Drogon::Drogon nlohmann_json::nlohmann_json)

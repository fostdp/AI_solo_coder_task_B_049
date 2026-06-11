#pragma once
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/daily_file_sink.h>
#include <spdlog/async.h>
#include <memory>
#include <string>

namespace tcm {

inline std::shared_ptr<spdlog::logger> create_logger(const std::string& name) {
    static bool registry_initialized = false;
    if (!registry_initialized) {
        spdlog::init_thread_pool(8192, 1);
        registry_initialized = true;
    }

    auto existing = spdlog::get(name);
    if (existing) return existing;

    std::vector<spdlog::sink_ptr> sinks;

    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_sink->set_level(spdlog::level::debug);
    console_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%n] [%^%l%$] [thread %t] %v");
    sinks.push_back(console_sink);

    try {
        auto file_sink = std::make_shared<spdlog::sinks::daily_file_sink_mt>(
            "logs/" + name + ".log", 23, 59);
        file_sink->set_level(spdlog::level::info);
        file_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%n] [%l] [thread %t] %v");
        sinks.push_back(file_sink);
    } catch (...) {}

    auto logger = std::make_shared<spdlog::async_logger>(
        name, sinks.begin(), sinks.end(),
        spdlog::thread_pool(), spdlog::async_overflow_policy::block);

    logger->set_level(spdlog::level::debug);
    logger->flush_on(spdlog::level::warn);
    logger->flush_every(std::chrono::seconds(5));
    spdlog::register_logger(logger);

    return logger;
}

#define TCM_LOG_TRACE(logger, ...)   logger->trace(__VA_ARGS__)
#define TCM_LOG_DEBUG(logger, ...)   logger->debug(__VA_ARGS__)
#define TCM_LOG_INFO(logger, ...)    logger->info(__VA_ARGS__)
#define TCM_LOG_WARN(logger, ...)    logger->warn(__VA_ARGS__)
#define TCM_LOG_ERROR(logger, ...)   logger->error(__VA_ARGS__)
#define TCM_LOG_CRITICAL(logger, ...) logger->critical(__VA_ARGS__)

#define TCM_LOG_FMT_TRACE(logger, fmt, ...)   logger->trace(fmt, __VA_ARGS__)
#define TCM_LOG_FMT_DEBUG(logger, fmt, ...)   logger->debug(fmt, __VA_ARGS__)
#define TCM_LOG_FMT_INFO(logger, fmt, ...)    logger->info(fmt, __VA_ARGS__)
#define TCM_LOG_FMT_WARN(logger, fmt, ...)    logger->warn(fmt, __VA_ARGS__)
#define TCM_LOG_FMT_ERROR(logger, fmt, ...)   logger->error(fmt, __VA_ARGS__)
#define TCM_LOG_FMT_CRITICAL(logger, fmt, ...) logger->critical(fmt, __VA_ARGS__)

}

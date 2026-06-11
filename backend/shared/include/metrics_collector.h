#pragma once
#include <string>
#include <sstream>
#include <mutex>
#include <chrono>
#include <cstdint>
#include <cstring>

namespace tcm {

struct Metrics {
    std::atomic<uint64_t> sensor_received{0};
    std::atomic<uint64_t> sensor_inserted{0};
    std::atomic<uint64_t> sensor_dropped{0};
    std::atomic<uint64_t> batch_count{0};
    std::atomic<uint64_t> batch_total_size{0};
    std::atomic<uint64_t> prediction_count{0};
    std::atomic<uint64_t> prediction_total_latency_us{0};
    std::atomic<uint64_t> alert_count{0};
    std::atomic<uint64_t> alert_conductance{0};
    std::atomic<uint64_t> alert_temperature{0};
    std::atomic<uint64_t> alert_emg{0};
    std::atomic<uint64_t> ble_packets_rx{0};
    std::atomic<uint64_t> ble_reconnects{0};
    std::atomic<uint64_t> ble_heartbeats{0};
    std::atomic<uint64_t> ble_offline_cache_size{0};
    std::atomic<uint64_t> ws_connections{0};
    std::atomic<uint64_t> ws_messages_sent{0};
    std::atomic<uint64_t> http_requests{0};
    std::atomic<int> queue_size{0};

    void record_sensor() { sensor_received.fetch_add(1, std::memory_order_relaxed); }
    void record_insert(uint64_t count = 1) { sensor_inserted.fetch_add(count, std::memory_order_relaxed); }
    void record_drop() { sensor_dropped.fetch_add(1, std::memory_order_relaxed); }
    void record_batch(uint64_t size) {
        batch_count.fetch_add(1, std::memory_order_relaxed);
        batch_total_size.fetch_add(size, std::memory_order_relaxed);
    }
    void record_prediction(uint64_t latency_us) {
        prediction_count.fetch_add(1, std::memory_order_relaxed);
        prediction_total_latency_us.fetch_add(latency_us, std::memory_order_relaxed);
    }
    void record_alert(const std::string& type) {
        alert_count.fetch_add(1, std::memory_order_relaxed);
        if (type.find("conductance") != std::string::npos) alert_conductance.fetch_add(1, std::memory_order_relaxed);
        else if (type.find("temperature") != std::string::npos) alert_temperature.fetch_add(1, std::memory_order_relaxed);
        else if (type.find("emg") != std::string::npos) alert_emg.fetch_add(1, std::memory_order_relaxed);
    }
    void record_ble_packet() { ble_packets_rx.fetch_add(1, std::memory_order_relaxed); }
    void record_ble_reconnect() { ble_reconnects.fetch_add(1, std::memory_order_relaxed); }
    void record_ble_heartbeat() { ble_heartbeats.fetch_add(1, std::memory_order_relaxed); }
    void record_ws_connect() { ws_connections.fetch_add(1, std::memory_order_relaxed); }
    void record_ws_disconnect() { ws_connections.fetch_sub(1, std::memory_order_relaxed); }
    void record_ws_broadcast() { ws_messages_sent.fetch_add(1, std::memory_order_relaxed); }
    void record_http_request() { http_requests.fetch_add(1, std::memory_order_relaxed); }
    void record_technique_analysis() { prediction_count.fetch_add(1, std::memory_order_relaxed); }
    void record_q_learning_update() { prediction_count.fetch_add(1, std::memory_order_relaxed); }

    std::string to_prometheus() const {
        std::ostringstream oss;
        auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

        oss << "# HELP tcm_sensor_received_total Total sensor data packets received\n";
        oss << "# TYPE tcm_sensor_received_total counter\n";
        oss << "tcm_sensor_received_total " << sensor_received.load() << "\n\n";

        oss << "# HELP tcm_sensor_inserted_total Total sensor data inserted to MongoDB\n";
        oss << "# TYPE tcm_sensor_inserted_total counter\n";
        oss << "tcm_sensor_inserted_total " << sensor_inserted.load() << "\n\n";

        oss << "# HELP tcm_sensor_dropped_total Total sensor data dropped (queue overflow)\n";
        oss << "# TYPE tcm_sensor_dropped_total counter\n";
        oss << "tcm_sensor_dropped_total " << sensor_dropped.load() << "\n\n";

        oss << "# HELP tcm_batch_count_total Total number of batch writes\n";
        oss << "# TYPE tcm_batch_count_total counter\n";
        oss << "tcm_batch_count_total " << batch_count.load() << "\n\n";

        oss << "# HELP tcm_batch_size_avg Average batch size\n";
        oss << "# TYPE tcm_batch_size_avg gauge\n";
        auto bc = batch_count.load();
        oss << "tcm_batch_size_avg " << (bc > 0 ? (double)batch_total_size.load() / bc : 0) << "\n\n";

        oss << "# HELP tcm_prediction_count_total Total prediction requests\n";
        oss << "# TYPE tcm_prediction_count_total counter\n";
        oss << "tcm_prediction_count_total " << prediction_count.load() << "\n\n";

        oss << "# HELP tcm_prediction_latency_avg_us Average prediction latency in microseconds\n";
        oss << "# TYPE tcm_prediction_latency_avg_us gauge\n";
        auto pc = prediction_count.load();
        oss << "tcm_prediction_latency_avg_us " << (pc > 0 ? (double)prediction_total_latency_us.load() / pc : 0) << "\n\n";

        oss << "# HELP tcm_alert_count_total Total alerts triggered\n";
        oss << "# TYPE tcm_alert_count_total counter\n";
        oss << "tcm_alert_count_total " << alert_count.load() << "\n";
        oss << "tcm_alert_count_total{type=\"conductance\"} " << alert_conductance.load() << "\n";
        oss << "tcm_alert_count_total{type=\"temperature\"} " << alert_temperature.load() << "\n";
        oss << "tcm_alert_count_total{type=\"emg\"} " << alert_emg.load() << "\n\n";

        oss << "# HELP tcm_ble_packets_rx_total Total BLE packets received\n";
        oss << "# TYPE tcm_ble_packets_rx_total counter\n";
        oss << "tcm_ble_packets_rx_total " << ble_packets_rx.load() << "\n\n";

        oss << "# HELP tcm_ble_reconnects_total Total BLE reconnection attempts\n";
        oss << "# TYPE tcm_ble_reconnects_total counter\n";
        oss << "tcm_ble_reconnects_total " << ble_reconnects.load() << "\n\n";

        oss << "# HELP tcm_ble_heartbeats_total Total BLE heartbeat packets\n";
        oss << "# TYPE tcm_ble_heartbeats_total counter\n";
        oss << "tcm_ble_heartbeats_total " << ble_heartbeats.load() << "\n\n";

        oss << "# HELP tcm_ble_offline_cache_size Current BLE offline cache size\n";
        oss << "# TYPE tcm_ble_offline_cache_size gauge\n";
        oss << "tcm_ble_offline_cache_size " << ble_offline_cache_size.load() << "\n\n";

        oss << "# HELP tcm_ws_connections_active Current WebSocket connections\n";
        oss << "# TYPE tcm_ws_connections_active gauge\n";
        oss << "tcm_ws_connections_active " << ws_connections.load() << "\n\n";

        oss << "# HELP tcm_ws_messages_sent_total Total WebSocket messages sent\n";
        oss << "# TYPE tcm_ws_messages_sent_total counter\n";
        oss << "tcm_ws_messages_sent_total " << ws_messages_sent.load() << "\n\n";

        oss << "# HELP tcm_http_requests_total Total HTTP requests\n";
        oss << "# TYPE tcm_http_requests_total counter\n";
        oss << "tcm_http_requests_total " << http_requests.load() << "\n\n";

        oss << "# HELP tcm_queue_size_current Current batch queue size\n";
        oss << "# TYPE tcm_queue_size_current gauge\n";
        oss << "tcm_queue_size_current " << queue_size.load() << "\n\n";

        return oss.str();
    }

    std::string to_json() const {
        std::ostringstream oss;
        oss << "{";
        oss << "\"sensor_received\":" << sensor_received.load() << ",";
        oss << "\"sensor_inserted\":" << sensor_inserted.load() << ",";
        oss << "\"sensor_dropped\":" << sensor_dropped.load() << ",";
        oss << "\"batch_count\":" << batch_count.load() << ",";
        oss << "\"avg_batch_size\":" << (batch_count.load() > 0 ? (double)batch_total_size.load() / batch_count.load() : 0) << ",";
        oss << "\"prediction_count\":" << prediction_count.load() << ",";
        oss << "\"avg_prediction_latency_us\":" << (prediction_count.load() > 0 ? (double)prediction_total_latency_us.load() / prediction_count.load() : 0) << ",";
        oss << "\"alert_count\":" << alert_count.load() << ",";
        oss << "\"alert_conductance\":" << alert_conductance.load() << ",";
        oss << "\"alert_temperature\":" << alert_temperature.load() << ",";
        oss << "\"alert_emg\":" << alert_emg.load() << ",";
        oss << "\"ble_packets_rx\":" << ble_packets_rx.load() << ",";
        oss << "\"ble_reconnects\":" << ble_reconnects.load() << ",";
        oss << "\"ws_connections\":" << ws_connections.load() << ",";
        oss << "\"ws_messages_sent\":" << ws_messages_sent.load() << ",";
        oss << "\"http_requests\":" << http_requests.load() << ",";
        oss << "\"queue_size\":" << queue_size.load();
        oss << "}";
        return oss.str();
    }
};

inline Metrics& global_metrics() {
    static Metrics m;
    return m;
}

}

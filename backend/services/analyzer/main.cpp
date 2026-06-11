#define CROW_MAIN
#include "crow.h"
#include "config.h"
#include "data_types.h"
#include "apriori_recommender.h"
#include "needle_technique_classifier.h"
#include "meridian_energy_balancer.h"
#include "q_learning_advisor.h"
#include "mongodb_manager.h"
#include "logger.h"
#include "metrics_collector.h"
#include <iostream>
#include <sstream>
#include <mutex>
#include <thread>
#include <chrono>

namespace tcm {

struct AnalyzerContext {
    AprioriRecommender* apriori;
    NeedleTechniqueClassifier* technique_classifier;
    MeridianEnergyBalancer* energy_balancer;
    QLearningAdvisor* q_learning;
    MongoDBManager* db;
    std::mutex mutex;
};

static AnalyzerContext* g_ctx = nullptr;

static std::string meridian_state_to_string(MeridianState s) {
    switch (s) {
        case MeridianState::BALANCED: return "balanced";
        case MeridianState::EXCESS: return "excess";
        case MeridianState::DEFICIENT: return "deficient";
        case MeridianState::HYPERACTIVE: return "hyperactive";
        case MeridianState::HYPOACTIVE: return "hypoactive";
        default: return "unknown";
    }
}

static std::string energy_to_json(const MeridianEnergy& me) {
    std::ostringstream oss;
    oss << "{"
        << "\"meridian_id\":\"" << me.meridian_id << "\","
        << "\"meridian_name\":\"" << me.meridian_name << "\","
        << "\"conductance_avg\":" << me.conductance_avg << ","
        << "\"temperature_avg\":" << me.temperature_avg << ","
        << "\"emg_activity\":" << me.emg_activity << ","
        << "\"energy_score\":" << me.energy_score << ","
        << "\"normalized_score\":" << me.normalized_score << ","
        << "\"state\":\"" << meridian_state_to_string(me.state) << "\","
        << "\"deviation_percent\":" << me.deviation_percent << ","
        << "\"sample_count\":" << me.sample_count
        << "}";
    return oss.str();
}

static std::string rule_to_json(const AssociationRule& r) {
    std::ostringstream oss;
    oss << "{"
        << "\"antecedent\":[";
    for (size_t i = 0; i < r.antecedent.size(); ++i) {
        if (i > 0) oss << ",";
        oss << "\"" << r.antecedent[i] << "\"";
    }
    oss << "],\"consequent\":[";
    for (size_t i = 0; i < r.consequent.size(); ++i) {
        if (i > 0) oss << ",";
        oss << "\"" << r.consequent[i] << "\"";
    }
    oss << "],\"support\":" << r.support
        << ",\"confidence\":" << r.confidence
        << ",\"lift\":" << r.lift
        << ",\"avg_efficacy\":" << r.avg_efficacy
        << "}";
    return oss.str();
}

static std::string combo_to_json(const AcupointCombination& c) {
    std::ostringstream oss;
    oss << "{"
        << "\"acupoints\":[";
    for (size_t i = 0; i < c.acupoints.size(); ++i) {
        if (i > 0) oss << ",";
        oss << "\"" << c.acupoints[i] << "\"";
    }
    oss << "],\"support\":" << c.support
        << ",\"avg_pain_relief\":" << c.avg_pain_relief
        << ",\"avg_deqi\":" << c.avg_deqi
        << ",\"sample_count\":" << c.sample_count
        << "}";
    return oss.str();
}

static std::string technique_result_to_json(const TechniqueAnalysisResult& r) {
    std::ostringstream oss;
    oss << "{"
        << "\"technique\":\"" << r.technique_name << "\","
        << "\"confidence\":" << r.confidence << ","
        << "\"is_active\":" << (r.is_active ? "true" : "false") << ","
        << "\"lifting_frequency_hz\":" << r.lifting_frequency_hz << ","
        << "\"twirling_frequency_hz\":" << r.twirling_frequency_hz << ","
        << "\"amplitude_ratio\":" << r.amplitude_ratio << ","
        << "\"window_start\":" << r.window_start << ","
        << "\"window_end\":" << r.window_end << ","
        << "\"features\":{"
        << "\"rms\":" << r.features.rms
        << ",\"mav\":" << r.features.mav
        << ",\"zero_crossing_rate\":" << r.features.zero_crossing_rate
        << ",\"peak_frequency\":" << r.features.peak_frequency
        << ",\"mean_frequency\":" << r.features.mean_frequency
        << ",\"spectral_entropy\":" << r.features.spectral_entropy
        << ",\"waveform_length\":" << r.features.waveform_length
        << ",\"variance\":" << r.features.variance
        << ",\"band_power_low\":" << r.features.band_power_low
        << ",\"band_power_mid\":" << r.features.band_power_mid
        << ",\"band_power_high\":" << r.features.band_power_high
        << "}}";
    return oss.str();
}

static std::string action_to_json(const AcupunctureAction& a) {
    std::ostringstream oss;
    oss << "{"
        << "\"needle_retention_min\":" << a.needle_retention_min << ","
        << "\"stimulation_frequency_hz\":" << a.stimulation_frequency_hz << ","
        << "\"needle_depth_mm\":" << a.needle_depth_mm << ","
        << "\"technique\":\"" << a.technique << "\","
        << "\"action_index\":" << a.action_index
        << "}";
    return oss.str();
}

} // namespace tcm

int main(int argc, char* argv[]) {
    using namespace tcm;

    std::string config_path = "./config.yaml";
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) config_path = argv[++i];
    }

    auto cfg = AppConfig::load_or_default(config_path);
    auto log = create_logger("analyzer");

    AprioriRecommender apriori;
    NeedleTechniqueClassifier technique;
    MeridianEnergyBalancer balancer;
    QLearningAdvisor q_learner;
    MongoDBManager db;

    AnalyzerContext ctx;
    ctx.apriori = &apriori;
    ctx.technique_classifier = &technique;
    ctx.energy_balancer = &balancer;
    ctx.q_learning = &q_learner;
    ctx.db = &db;
    g_ctx = &ctx;

    TCM_LOG_INFO(log, "Connecting MongoDB at {}", cfg.mongodb.uri);
    db.initialize(cfg.mongodb.uri, cfg.mongodb.db_name);

    technique.initialize(1000, 500);
    balancer.initialize(300.0);
    q_learner.initialize(0.001, 0.01, 0.95, 0.2);

    TCM_LOG_INFO(log, "Loading historical data for Apriori...");
    {
        auto records = db.get_all_efficacy_records(1000);
        std::map<std::string, std::vector<std::string>> session_acupoints;
        std::map<std::string, double> session_pain_relief;
        std::map<std::string, double> session_deqi;

        for (const auto& r : records) {
            session_acupoints[r.session_id].push_back(r.acupoint_id);
            session_pain_relief[r.session_id] = r.pain_relief_rate;
            session_deqi[r.session_id] = r.deqi_intensity;
        }

        for (const auto& kv : session_acupoints) {
            apriori.add_transaction(kv.second,
                session_pain_relief[kv.first],
                session_deqi[kv.first]);
        }

        TCM_LOG_INFO(log, "Loaded {} sessions for Apriori", session_acupoints.size());

        if (apriori.transaction_count() > 5) {
            apriori.run_apriori(0.03, 0.25, 1.2, 4);
            TCM_LOG_INFO(log, "Apriori generated {} rules", apriori.get_rules(1000).size());
        }
    }

    crow::SimpleApp app;

    CROW_ROUTE(app, "/api/health")([]() {
        crow::json::wvalue r;
        r["status"] = "ok";
        r["service"] = "analyzer";
        return r;
    });

    CROW_ROUTE(app, "/metrics")([]() {
        return crow::response(200, "text/plain", global_metrics().to_prometheus());
    });

    CROW_ROUTE(app, "/api/association/rules")
    .methods(crow::HTTPMethod::GET)([&apriori](const crow::request& req) {
        int top_k = 20;
        auto top_param = req.url_params.get("top");
        if (top_param) top_k = std::stoi(top_param);

        auto rules = apriori.get_rules(top_k);
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < rules.size(); ++i) {
            if (i > 0) oss << ",";
            oss << rule_to_json(rules[i]);
        }
        oss << "]";
        return crow::response(200, "application/json", oss.str());
    });

    CROW_ROUTE(app, "/api/association/itemsets")
    .methods(crow::HTTPMethod::GET)([&apriori](const crow::request& req) {
        int min_count = 5;
        auto mc_param = req.url_params.get("min_count");
        if (mc_param) min_count = std::stoi(mc_param);

        auto itemsets = apriori.get_frequent_itemsets(min_count);
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < itemsets.size(); ++i) {
            if (i > 0) oss << ",";
            oss << combo_to_json(itemsets[i]);
        }
        oss << "]";
        return crow::response(200, "application/json", oss.str());
    });

    CROW_ROUTE(app, "/api/association/recommend")
    .methods(crow::HTTPMethod::POST)([&apriori](const crow::request& req) {
        auto body = crow::json::load(req.body);
        if (!body) return crow::response(400, "{\"error\":\"invalid json\"}");

        std::vector<std::string> current_acupoints;
        if (body.has("acupoints")) {
            for (const auto& ap : body["acupoints"]) {
                current_acupoints.push_back(std::string(ap.s()));
            }
        }

        int top_k = 10;
        if (body.has("top_k")) top_k = (int)body["top_k"].i();

        auto recs = apriori.get_recommendations(current_acupoints, top_k);
        std::ostringstream oss;
        oss << "{"
            << "\"current_acupoints\":[";
        for (size_t i = 0; i < current_acupoints.size(); ++i) {
            if (i > 0) oss << ",";
            oss << "\"" << current_acupoints[i] << "\"";
        }
        oss << "],\"recommendations\":[";
        for (size_t i = 0; i < recs.size(); ++i) {
            if (i > 0) oss << ",";
            oss << combo_to_json(recs[i]);
        }
        oss << "],\"total_transactions\":" << apriori.transaction_count()
            << "}";
        return crow::response(200, "application/json", oss.str());
    });

    CROW_ROUTE(app, "/api/technique/analyze")
    .methods(crow::HTTPMethod::POST)([&technique](const crow::request& req) {
        auto body = crow::json::load(req.body);
        if (!body) return crow::response(400, "{\"error\":\"invalid json\"}");

        std::vector<double> emg_signal;
        if (body.has("emg_signal")) {
            for (const auto& v : body["emg_signal"]) {
                emg_signal.push_back(v.d());
            }
        }

        uint64_t timestamp = 0;
        if (body.has("timestamp")) timestamp = (uint64_t)body["timestamp"].i();

        auto result = technique.analyze(emg_signal, timestamp);
        global_metrics().record_technique_analysis();
        return crow::response(200, "application/json", technique_result_to_json(result));
    });

    CROW_ROUTE(app, "/api/technique/types")([]() {
        std::ostringstream oss;
        oss << "[";
        const char* types[] = {
            "resting", "lifting_thrusting", "twirling",
            "reinforcing", "reducing", "even_method"
        };
        const char* names[] = {
            "静息", "提插法", "捻转法",
            "补法", "泻法", "平补平泻"
        };
        for (int i = 0; i < 6; ++i) {
            if (i > 0) oss << ",";
            oss << "{\"id\":\"" << types[i] << "\",\"name\":\"" << names[i] << "\"}";
        }
        oss << "]";
        return crow::response(200, "application/json", oss.str());
    });

    CROW_ROUTE(app, "/api/energy/balance")
    .methods(crow::HTTPMethod::GET)([&balancer, &log](const crow::request& req) {
        auto vol_param = req.url_params.get("volunteer_id");
        std::string volunteer_id = vol_param ? std::string(vol_param) : "";

        (void)volunteer_id;

        auto now = std::chrono::system_clock::now();
        uint64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()).count();

        auto balance = balancer.compute_balance(now_ms);

        std::ostringstream oss;
        oss << "{"
            << "\"overall_balance_score\":" << balance.overall_balance_score << ","
            << "\"yin_energy\":" << balance.yin_energy << ","
            << "\"yang_energy\":" << balance.yang_energy << ","
            << "\"yin_yang_ratio\":" << balance.yin_yang_ratio << ","
            << "\"dominant_meridian\":\"" << balance.dominant_meridian << "\","
            << "\"weakest_meridian\":\"" << balance.weakest_meridian << "\","
            << "\"excess_meridians\":[";
        for (size_t i = 0; i < balance.excess_meridians.size(); ++i) {
            if (i > 0) oss << ",";
            oss << "\"" << balance.excess_meridians[i] << "\"";
        }
        oss << "],\"deficient_meridians\":[";
        for (size_t i = 0; i < balance.deficient_meridians.size(); ++i) {
            if (i > 0) oss << ",";
            oss << "\"" << balance.deficient_meridians[i] << "\"";
        }
        oss << "],\"meridians\":[";
        for (size_t i = 0; i < balance.meridians.size(); ++i) {
            if (i > 0) oss << ",";
            oss << energy_to_json(balance.meridians[i]);
        }
        oss << "],\"timestamp\":" << balance.timestamp
            << "}";
        return crow::response(200, "application/json", oss.str());
    });

    CROW_ROUTE(app, "/api/energy/five_elements")
    .methods(crow::HTTPMethod::GET)([&balancer](const crow::request& req) {
        auto now = std::chrono::system_clock::now();
        uint64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()).count();

        auto balance = balancer.compute_balance(now_ms);
        auto elements = balancer.compute_five_element_energy(balance);

        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < elements.size(); ++i) {
            if (i > 0) oss << ",";
            oss << "{"
                << "\"element\":\"" << elements[i].element << "\","
                << "\"energy\":" << elements[i].energy << ","
                << "\"meridians\":[";
            for (size_t j = 0; j < elements[i].meridians.size(); ++j) {
                if (j > 0) oss << ",";
                oss << "\"" << elements[i].meridians[j] << "\"";
            }
            oss << "]}";
        }
        oss << "]";
        return crow::response(200, "application/json", oss.str());
    });

    CROW_ROUTE(app, "/api/energy/sensor")
    .methods(crow::HTTPMethod::POST)([&balancer](const crow::request& req) {
        auto body = crow::json::load(req.body);
        if (!body) return crow::response(400, "{\"error\":\"invalid json\"}");

        std::string meridian_id = std::string(body["meridian_id"].s());
        std::string acupoint_id = std::string(body["acupoint_id"].s());
        double sc = body["skin_conductance"].d();
        double temp = body["infrared_temperature"].d();
        double emg = body["emg_amplitude"].d();
        uint64_t ts = (uint64_t)body["timestamp"].i();

        balancer.add_sensor_data(meridian_id, acupoint_id, sc, temp, emg, ts);

        return crow::response(200, "{\"status\":\"ok\"}");
    });

    CROW_ROUTE(app, "/api/qlearn/recommend")
    .methods(crow::HTTPMethod::POST)([&q_learner](const crow::request& req) {
        auto body = crow::json::load(req.body);
        if (!body) return crow::response(400, "{\"error\":\"invalid json\"}");

        AcupunctureState state;
        state.volunteer_id = std::string(body["volunteer_id"].s());
        state.meridian_id = body.has("meridian_id") ? std::string(body["meridian_id"].s()) : "";
        state.deqi_intensity = body.has("deqi_intensity") ? body["deqi_intensity"].d() : 0.5;
        state.pain_level = body.has("pain_level") ? body["pain_level"].d() : 0.5;
        state.current_duration_min = body.has("current_duration_min") ? body["current_duration_min"].d() : 0;
        state.session_count = body.has("session_count") ? (int)body["session_count"].i() : 0;
        state.body_region = body.has("body_region") ? std::string(body["body_region"].s()) : "";

        if (body.has("acupoints")) {
            for (const auto& ap : body["acupoints"]) {
                state.acupoints.push_back(std::string(ap.s()));
            }
        }

        auto result = q_learner.recommend_action(state);

        std::ostringstream oss;
        oss << "{"
            << "\"recommended_action\":" << action_to_json(result.recommended_action) << ","
            << "\"expected_reward\":" << result.expected_reward << ","
            << "\"confidence\":" << result.confidence << ","
            << "\"is_exploration\":" << (result.is_exploration ? "true" : "false") << ","
            << "\"top_actions\":[";
        for (size_t i = 0; i < result.top_actions.size(); ++i) {
            if (i > 0) oss << ",";
            oss << "{"
                << "\"action\":" << action_to_json(result.top_actions[i]) << ","
                << "\"value\":" << result.action_values[i]
                << "}";
        }
        oss << "]}";
        return crow::response(200, "application/json", oss.str());
    });

    CROW_ROUTE(app, "/api/qlearn/feedback")
    .methods(crow::HTTPMethod::POST)([&q_learner](const crow::request& req) {
        auto body = crow::json::load(req.body);
        if (!body) return crow::response(400, "{\"error\":\"invalid json\"}");

        AcupunctureState state;
        state.volunteer_id = std::string(body["volunteer_id"].s());
        state.meridian_id = body.has("meridian_id") ? std::string(body["meridian_id"].s()) : "";
        state.deqi_intensity = body.has("deqi_intensity") ? body["deqi_intensity"].d() : 0.5;
        state.pain_level = body.has("pain_level") ? body["pain_level"].d() : 0.5;
        state.current_duration_min = 0;

        AcupunctureAction action;
        action.needle_retention_min = body["action"]["needle_retention_min"].d();
        action.stimulation_frequency_hz = body["action"]["stimulation_frequency_hz"].d();
        action.needle_depth_mm = body["action"]["needle_depth_mm"].d();
        action.technique = std::string(body["action"]["technique"].s());
        action.action_index = (int)body["action"]["action_index"].i();

        double reward = body["reward"].d();

        AcupunctureState next_state = state;
        if (body.has("next_state")) {
            next_state.deqi_intensity = body["next_state"]["deqi_intensity"].d();
            next_state.pain_level = body["next_state"]["pain_level"].d();
            next_state.current_duration_min = action.needle_retention_min;
        }

        bool is_terminal = body.has("is_terminal") ? (bool)body["is_terminal"].b() : true;

        q_learner.record_result(state, action, reward, next_state, is_terminal);
        q_learner.decay_exploration();

        std::ostringstream oss;
        oss << "{"
            << "\"status\":\"ok\","
            << "\"total_updates\":" << q_learner.get_total_updates() << ","
            << "\"exploration_rate\":" << q_learner.get_exploration_rate() << ","
            << "\"state_count\":" << q_learner.get_state_count()
            << "}";
        return crow::response(200, "application/json", oss.str());
    });

    CROW_ROUTE(app, "/api/qlearn/stats")([&q_learner]() {
        std::ostringstream oss;
        oss << "{"
            << "\"state_count\":" << q_learner.get_state_count() << ","
            << "\"total_updates\":" << q_learner.get_total_updates() << ","
            << "\"exploration_rate\":" << q_learner.get_exploration_rate() << ","
            << "\"average_reward\":" << q_learner.get_average_reward() << ","
            << "\"action_count\":" << q_learner.get_action_count()
            << "}";
        return crow::response(200, "application/json", oss.str());
    });

    CROW_ROUTE(app, "/api/qlearn/actions")([&q_learner]() {
        auto actions = q_learner.get_all_actions();
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < actions.size(); ++i) {
            if (i > 0) oss << ",";
            oss << action_to_json(actions[i]);
        }
        oss << "]";
        return crow::response(200, "application/json", oss.str());
    });

    CROW_ROUTE(app, "/api/sensor/forward")
    .methods(crow::HTTPMethod::POST)([&balancer, &technique](const crow::request& req) {
        auto body = crow::json::load(req.body);
        if (!body) return crow::response(400, "{\"error\":\"invalid json\"}");

        std::string mid = body.has("meridian_id") ? std::string(body["meridian_id"].s()) : "";
        std::string aid = body.has("acupoint_id") ? std::string(body["acupoint_id"].s()) : "";
        double sc = body.has("skin_conductance") ? body["skin_conductance"].d() : 0;
        double temp = body.has("infrared_temperature") ? body["infrared_temperature"].d() : 0;
        double emg = body.has("emg_amplitude") ? body["emg_amplitude"].d() : 0;
        uint64_t ts = body.has("timestamp") ? (uint64_t)body["timestamp"].i() : 0;

        if (!mid.empty()) {
            balancer.add_sensor_data(mid, aid, sc, temp, emg, ts);
        }

        return crow::response(200, "{\"status\":\"ok\"}");
    });

    int port = cfg.analyzer.http_port;
    TCM_LOG_INFO(log, "HTTP on port {}", port);
    app.loglevel(crow::LogLevel::Warning).port(port).multithreaded().run();

    db.shutdown();
    return 0;
}

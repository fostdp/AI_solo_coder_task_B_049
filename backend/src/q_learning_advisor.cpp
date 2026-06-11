#include "q_learning_advisor.h"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <fstream>
#include <iostream>

namespace tcm {

QLearningAdvisor::QLearningAdvisor()
    : learning_rate_(0.1)
    , discount_factor_(0.95)
    , exploration_rate_(0.2)
    , exploration_decay_(0.995)
    , min_exploration_rate_(0.05)
    , total_updates_(0)
    , total_reward_(0)
    , reward_count_(0)
    , rng_(42) {
    init_default_actions();
}

void QLearningAdvisor::initialize(double learning_rate,
                                  double discount_factor,
                                  double exploration_rate,
                                  double exploration_decay) {
    learning_rate_ = learning_rate;
    discount_factor_ = discount_factor;
    exploration_rate_ = exploration_rate;
    exploration_decay_ = exploration_decay;
    total_updates_ = 0;
    total_reward_ = 0;
    reward_count_ = 0;
}

void QLearningAdvisor::init_default_actions() {
    actions_.clear();

    std::vector<double> durations = {10.0, 15.0, 20.0, 25.0, 30.0, 40.0};
    std::vector<double> frequencies = {0.5, 1.0, 2.0, 3.0};
    std::vector<double> depths = {5.0, 10.0, 15.0, 20.0};

    for (size_t i = 0; i < durations.size(); ++i) {
        AcupunctureAction a;
        a.needle_retention_min = durations[i];
        a.stimulation_frequency_hz = 1.5;
        a.needle_depth_mm = 12.0;
        a.technique = "balanced";
        a.action_index = (int)i;
        actions_.push_back(a);
    }

    for (size_t i = 0; i < durations.size(); ++i) {
        AcupunctureAction a;
        a.needle_retention_min = durations[i];
        a.stimulation_frequency_hz = 2.5;
        a.needle_depth_mm = 15.0;
        a.technique = "reinforcing";
        a.action_index = (int)(durations.size() + i);
        actions_.push_back(a);
    }

    for (size_t i = 0; i < durations.size(); ++i) {
        AcupunctureAction a;
        a.needle_retention_min = durations[i];
        a.stimulation_frequency_hz = 3.0;
        a.needle_depth_mm = 10.0;
        a.technique = "reducing";
        a.action_index = (int)(2 * durations.size() + i);
        actions_.push_back(a);
    }
}

std::string QLearningAdvisor::discretize_deqi(double deqi) const {
    if (deqi < 0.2) return "VLOW";
    if (deqi < 0.4) return "LOW";
    if (deqi < 0.6) return "MED";
    if (deqi < 0.8) return "HIGH";
    return "VHIGH";
}

std::string QLearningAdvisor::discretize_pain(double pain) const {
    if (pain < 0.2) return "VLOW";
    if (pain < 0.4) return "LOW";
    if (pain < 0.6) return "MED";
    if (pain < 0.8) return "HIGH";
    return "VHIGH";
}

std::string QLearningAdvisor::discretize_duration(double duration) const {
    if (duration < 10) return "SHORT";
    if (duration < 20) return "MED";
    if (duration < 30) return "LONG";
    return "VLONG";
}

std::string QLearningAdvisor::state_to_key(const AcupunctureState& state) const {
    std::ostringstream oss;
    oss << state.volunteer_id << "|"
        << state.meridian_id << "|"
        << discretize_deqi(state.deqi_intensity) << "|"
        << discretize_pain(state.pain_level) << "|"
        << discretize_duration(state.current_duration_min);

    if (!state.acupoints.empty()) {
        oss << "|" << state.acupoints[0];
    }

    return oss.str();
}

void QLearningAdvisor::ensure_state_exists(const std::string& state_key) {
    if (q_table_.find(state_key) == q_table_.end()) {
        q_table_[state_key] = std::vector<double>(actions_.size(), 0.0);
        state_visit_count_[state_key] = 0;
    }
}

std::vector<double> QLearningAdvisor::get_q_values(const std::string& state_key) {
    ensure_state_exists(state_key);
    return q_table_[state_key];
}

int QLearningAdvisor::select_best_action(const std::vector<double>& q_values) const {
    int best_idx = 0;
    double best_val = q_values[0];
    for (size_t i = 1; i < q_values.size(); ++i) {
        if (q_values[i] > best_val) {
            best_val = q_values[i];
            best_idx = (int)i;
        }
    }
    return best_idx;
}

int QLearningAdvisor::select_action_epsilon_greedy(const std::vector<double>& q_values) {
    std::uniform_real_distribution<> dist(0.0, 1.0);

    if (dist(rng_) < exploration_rate_) {
        std::uniform_int_distribution<> int_dist(0, (int)q_values.size() - 1);
        return int_dist(rng_);
    }

    return select_best_action(q_values);
}

QLearningResult QLearningAdvisor::recommend_action(const AcupunctureState& state) {
    std::string state_key = state_to_key(state);
    auto q_values = get_q_values(state_key);

    int action_idx = select_action_epsilon_greedy(q_values);
    bool is_exploration = (action_idx != select_best_action(q_values));

    double max_q = *std::max_element(q_values.begin(), q_values.end());
    double min_q = *std::min_element(q_values.begin(), q_values.end());
    double range = max_q - min_q;
    double confidence = range > 0 ? (q_values[action_idx] - min_q) / range : 0.5;

    QLearningResult result;
    result.recommended_action = actions_[action_idx];
    result.expected_reward = q_values[action_idx];
    result.confidence = confidence;
    result.is_exploration = is_exploration;
    result.action_values = q_values;

    std::vector<int> indices(q_values.size());
    for (size_t i = 0; i < indices.size(); ++i) indices[i] = (int)i;
    std::sort(indices.begin(), indices.end(),
        [&q_values](int a, int b) { return q_values[a] > q_values[b]; });

    int top_n = std::min(5, (int)q_values.size());
    for (int i = 0; i < top_n; ++i) {
        result.top_actions.push_back(actions_[indices[i]]);
    }

    return result;
}

void QLearningAdvisor::update_q_table(const TrainingEpisode& episode) {
    ensure_state_exists(episode.state_key);
    ensure_state_exists(episode.next_state_key);

    auto& current_q = q_table_[episode.state_key];
    auto& next_q = q_table_[episode.next_state_key];

    state_visit_count_[episode.state_key]++;

    double old_value = current_q[episode.action_taken];
    double next_max = *std::max_element(next_q.begin(), next_q.end());

    double target = episode.reward;
    if (!episode.done) {
        target += discount_factor_ * next_max;
    }

    double new_value = old_value + learning_rate_ * (target - old_value);
    current_q[episode.action_taken] = new_value;

    total_updates_++;
    total_reward_ += episode.reward;
    reward_count_++;
}

void QLearningAdvisor::record_result(const AcupunctureState& state,
                                     const AcupunctureAction& action,
                                     double reward,
                                     const AcupunctureState& next_state,
                                     bool is_terminal) {
    TrainingEpisode episode;
    episode.volunteer_id = state.volunteer_id;
    episode.state_key = state_to_key(state);
    episode.action_taken = action.action_index;
    episode.reward = reward;
    episode.next_state_key = state_to_key(next_state);
    episode.done = is_terminal;

    update_q_table(episode);
}

double QLearningAdvisor::compute_reward(const AcupunctureState& state,
                                        const AcupunctureAction& action,
                                        const AcupunctureState& next_state) const {
    (void)state;
    (void)action;

    double deqi_improvement = next_state.deqi_intensity - state.deqi_intensity;
    double pain_reduction = state.pain_level - next_state.pain_level;

    double reward = deqi_improvement * 50.0 + pain_reduction * 30.0;

    if (next_state.deqi_intensity > 0.7 && next_state.pain_level < 0.3) {
        reward += 20.0;
    }

    double time_penalty = action.needle_retention_min * 0.2;
    reward -= time_penalty;

    return reward;
}

void QLearningAdvisor::decay_exploration() {
    exploration_rate_ *= exploration_decay_;
    if (exploration_rate_ < min_exploration_rate_) {
        exploration_rate_ = min_exploration_rate_;
    }
}

std::vector<AcupunctureAction> QLearningAdvisor::get_all_actions() const {
    return actions_;
}

void QLearningAdvisor::reset_q_table() {
    q_table_.clear();
    state_visit_count_.clear();
    total_updates_ = 0;
    total_reward_ = 0;
    reward_count_ = 0;
}

AcupunctureAction QLearningAdvisor::action_from_index(int index) {
    QLearningAdvisor dummy;
    dummy.init_default_actions();
    if (index >= 0 && index < (int)dummy.actions_.size()) {
        return dummy.actions_[index];
    }
    return AcupunctureAction();
}

int QLearningAdvisor::action_to_index(const AcupunctureAction& action) {
    return action.action_index;
}

void QLearningAdvisor::save_model(const std::string& path) const {
    std::ofstream f(path);
    if (!f.is_open()) return;

    f << "Q_TABLE_V1\n";
    f << q_table_.size() << "\n";
    for (const auto& kv : q_table_) {
        f << kv.first << "\n";
        f << kv.second.size() << "\n";
        for (double v : kv.second) {
            f << v << " ";
        }
        f << "\n";
    }
    f.close();
}

void QLearningAdvisor::load_model(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) return;

    std::string header;
    f >> header;
    if (header != "Q_TABLE_V1") {
        f.close();
        return;
    }

    size_t num_states;
    f >> num_states;

    q_table_.clear();
    for (size_t i = 0; i < num_states; ++i) {
        std::string key;
        size_t num_actions;
        f >> key >> num_actions;

        std::vector<double> values(num_actions);
        for (size_t j = 0; j < num_actions; ++j) {
            f >> values[j];
        }
        q_table_[key] = values;
    }
    f.close();
}

} // namespace tcm

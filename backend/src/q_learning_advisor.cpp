#include "q_learning_advisor.h"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <fstream>
#include <iostream>
#include <numeric>

namespace tcm {

QLearningAdvisor::QLearningAdvisor()
    : actor_learning_rate_(0.001)
    , critic_learning_rate_(0.01)
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

void QLearningAdvisor::initialize(double actor_lr,
                                  double critic_lr,
                                  double discount_factor,
                                  double exploration_rate) {
    actor_learning_rate_ = actor_lr;
    critic_learning_rate_ = critic_lr;
    discount_factor_ = discount_factor;
    exploration_rate_ = exploration_rate;
    total_updates_ = 0;
    total_reward_ = 0;
    reward_count_ = 0;
}

void QLearningAdvisor::init_default_actions() {
    actions_.clear();

    std::vector<double> durations = {10.0, 15.0, 20.0, 25.0, 30.0, 40.0};

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
    if (actor_weights_.find(state_key) == actor_weights_.end()) {
        actor_weights_[state_key] = std::vector<double>(actions_.size(), 0.0);
        critic_values_[state_key] = 0.0;
    }
}

std::vector<double> QLearningAdvisor::get_action_preferences(const std::string& state_key) {
    ensure_state_exists(state_key);
    return actor_weights_[state_key];
}

std::vector<double> QLearningAdvisor::compute_softmax(const std::vector<double>& preferences) const {
    std::vector<double> probs(preferences.size());
    double max_pref = *std::max_element(preferences.begin(), preferences.end());
    double sum_exp = 0;
    for (size_t i = 0; i < preferences.size(); ++i) {
        probs[i] = std::exp(preferences[i] - max_pref);
        sum_exp += probs[i];
    }
    if (sum_exp > 0) {
        for (auto& p : probs) p /= sum_exp;
    } else {
        double uniform = 1.0 / probs.size();
        for (auto& p : probs) p = uniform;
    }
    return probs;
}

int QLearningAdvisor::sample_action_from_probabilities(const std::vector<double>& probs) {
    std::uniform_real_distribution<> dist(0.0, 1.0);
    double r = dist(rng_);
    double cum = 0;
    for (size_t i = 0; i < probs.size(); ++i) {
        cum += probs[i];
        if (r <= cum) return (int)i;
    }
    return (int)probs.size() - 1;
}

int QLearningAdvisor::select_best_action(const std::vector<double>& probs) const {
    int best_idx = 0;
    double best = probs[0];
    for (size_t i = 1; i < probs.size(); ++i) {
        if (probs[i] > best) {
            best = probs[i];
            best_idx = (int)i;
        }
    }
    return best_idx;
}

QLearningResult QLearningAdvisor::recommend_action(const AcupunctureState& state) {
    std::string state_key = state_to_key(state);
    auto preferences = get_action_preferences(state_key);
    auto probs = compute_softmax(preferences);

    std::uniform_real_distribution<> dist(0.0, 1.0);
    bool is_exploration = dist(rng_) < exploration_rate_;

    int action_idx;
    if (is_exploration) {
        std::uniform_int_distribution<> int_dist(0, (int)probs.size() - 1);
        action_idx = int_dist(rng_);
    } else {
        action_idx = sample_action_from_probabilities(probs);
    }

    int greedy_idx = select_best_action(probs);
    if (!is_exploration) is_exploration = (action_idx != greedy_idx);

    double state_value = critic_values_.count(state_key) ? critic_values_[state_key] : 0.0;
    double confidence = probs[greedy_idx];

    QLearningResult result;
    result.recommended_action = actions_[action_idx];
    result.expected_reward = probs[action_idx];
    result.confidence = confidence;
    result.is_exploration = is_exploration;
    result.action_values = preferences;
    result.action_probabilities = probs;
    result.state_value = state_value;
    result.td_error = 0.0;

    std::vector<int> indices(probs.size());
    for (size_t i = 0; i < indices.size(); ++i) indices[i] = (int)i;
    std::sort(indices.begin(), indices.end(),
        [&probs](int a, int b) { return probs[a] > probs[b]; });

    int top_n = std::min(5, (int)probs.size());
    for (int i = 0; i < top_n; ++i) {
        result.top_actions.push_back(actions_[indices[i]]);
    }

    return result;
}

double QLearningAdvisor::normalize_reward(double raw_reward) const {
    double abs_r = std::fabs(raw_reward);
    if (abs_r < 1e-6) return 0.0;
    double scaled = std::tanh(raw_reward / 10.0);
    return scaled;
}

double QLearningAdvisor::compute_reward(const AcupunctureState& state,
                                        const AcupunctureAction& action,
                                        const AcupunctureState& next_state) const {
    (void)state;
    (void)action;

    double deqi_improvement = next_state.deqi_intensity - state.deqi_intensity;
    double pain_reduction = state.pain_level - next_state.pain_level;

    double reward = deqi_improvement * 2.0 + pain_reduction * 1.5;

    if (next_state.deqi_intensity > 0.7 && next_state.pain_level < 0.3) {
        reward += 1.0;
    }

    double time_penalty = action.needle_retention_min * 0.005;
    reward -= time_penalty;

    return normalize_reward(reward);
}

double QLearningAdvisor::compute_td_error(double reward, const std::string& state_key,
                                          const std::string& next_state_key, bool done) {
    ensure_state_exists(state_key);
    ensure_state_exists(next_state_key);

    double v_current = critic_values_[state_key];
    double v_next = done ? 0.0 : critic_values_[next_state_key];
    double norm_reward = normalize_reward(reward);
    double td_error = norm_reward + discount_factor_ * v_next - v_current;
    return td_error;
}

void QLearningAdvisor::update_actor(const std::string& state_key, int action_taken, double td_error) {
    ensure_state_exists(state_key);
    auto& weights = actor_weights_[state_key];
    auto probs = compute_softmax(weights);

    for (size_t i = 0; i < weights.size(); ++i) {
        double indicator = (i == (size_t)action_taken) ? 1.0 : 0.0;
        double grad_log_pi = indicator - probs[i];
        weights[i] += actor_learning_rate_ * td_error * grad_log_pi;
    }
}

void QLearningAdvisor::update_critic(const std::string& state_key, double td_error) {
    ensure_state_exists(state_key);
    critic_values_[state_key] += critic_learning_rate_ * td_error;
}

void QLearningAdvisor::update_q_table(const TrainingEpisode& episode) {
    ensure_state_exists(episode.state_key);
    ensure_state_exists(episode.next_state_key);

    double td_error = compute_td_error(episode.reward, episode.state_key,
                                       episode.next_state_key, episode.done);

    update_critic(episode.state_key, td_error);
    update_actor(episode.state_key, episode.action_taken, td_error);

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
    actor_weights_.clear();
    critic_values_.clear();
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

    f << "ACTOR_CRITIC_V1\n";
    f << actor_weights_.size() << "\n";
    for (const auto& kv : actor_weights_) {
        f << kv.first << "\n";
        f << "ACTOR " << kv.second.size() << "\n";
        for (double v : kv.second) f << v << " ";
        f << "\n";
        auto cit = critic_values_.find(kv.first);
        double cv = (cit != critic_values_.end()) ? cit->second : 0.0;
        f << "CRITIC " << cv << "\n";
    }
    f.close();
}

void QLearningAdvisor::load_model(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) return;

    std::string header;
    f >> header;
    if (header != "ACTOR_CRITIC_V1" && header != "Q_TABLE_V1") {
        f.close();
        return;
    }

    size_t num_states;
    f >> num_states;

    actor_weights_.clear();
    critic_values_.clear();

    for (size_t i = 0; i < num_states; ++i) {
        std::string key;
        f >> key;

        std::string type_tag;
        size_t num_actions;
        f >> type_tag >> num_actions;

        std::vector<double> values(num_actions);
        for (size_t j = 0; j < num_actions; ++j) {
            f >> values[j];
        }
        actor_weights_[key] = values;

        if (header == "ACTOR_CRITIC_V1") {
            std::string critic_tag;
            double cv;
            f >> critic_tag >> cv;
            critic_values_[key] = cv;
        } else {
            double mean_q = std::accumulate(values.begin(), values.end(), 0.0) / std::max(1, (int)values.size());
            critic_values_[key] = mean_q;
        }
    }
    f.close();
}

} // namespace tcm

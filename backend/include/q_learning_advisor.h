#pragma once
#include <string>
#include <vector>
#include <map>
#include <random>
#include <cstdint>

namespace tcm {

struct AcupunctureState {
    std::string volunteer_id;
    std::string meridian_id;
    std::vector<std::string> acupoints;
    double deqi_intensity;
    double pain_level;
    double skin_conductance_base;
    double current_duration_min;
    int session_count;
    std::string body_region;
};

struct AcupunctureAction {
    double needle_retention_min;
    double stimulation_frequency_hz;
    double needle_depth_mm;
    std::string technique;
    int action_index;
};

struct QLearningResult {
    AcupunctureAction recommended_action;
    double expected_reward;
    double confidence;
    std::vector<AcupunctureAction> top_actions;
    std::vector<double> action_values;
    bool is_exploration;
};

struct TrainingEpisode {
    std::string volunteer_id;
    std::string state_key;
    int action_taken;
    double reward;
    std::string next_state_key;
    bool done;
};

class QLearningAdvisor {
public:
    QLearningAdvisor();
    ~QLearningAdvisor() = default;

    void initialize(double learning_rate = 0.1,
                    double discount_factor = 0.95,
                    double exploration_rate = 0.2,
                    double exploration_decay = 0.995);

    QLearningResult recommend_action(const AcupunctureState& state);

    void record_result(const AcupunctureState& state,
                       const AcupunctureAction& action,
                       double reward,
                       const AcupunctureState& next_state,
                       bool is_terminal = false);

    void update_q_table(const TrainingEpisode& episode);

    void decay_exploration();
    void set_exploration_rate(double rate) { exploration_rate_ = rate; }
    double get_exploration_rate() const { return exploration_rate_; }

    std::vector<AcupunctureAction> get_all_actions() const;
    int get_action_count() const { return (int)actions_.size(); }

    size_t get_state_count() const { return q_table_.size(); }
    size_t get_total_updates() const { return total_updates_; }

    void load_model(const std::string& path);
    void save_model(const std::string& path) const;

    void reset_q_table();

    double get_average_reward() const { return total_reward_ / std::max(1, (int)reward_count_); }

    static AcupunctureAction action_from_index(int index);
    static int action_to_index(const AcupunctureAction& action);

private:
    double learning_rate_;
    double discount_factor_;
    double exploration_rate_;
    double exploration_decay_;
    double min_exploration_rate_;

    std::vector<AcupunctureAction> actions_;
    std::map<std::string, std::vector<double>> q_table_;
    std::map<std::string, int> state_visit_count_;

    size_t total_updates_;
    double total_reward_;
    size_t reward_count_;

    mutable std::mt19937 rng_;

    std::string state_to_key(const AcupunctureState& state) const;
    std::vector<double> get_q_values(const std::string& state_key);
    void ensure_state_exists(const std::string& state_key);

    int select_action_epsilon_greedy(const std::vector<double>& q_values);
    int select_best_action(const std::vector<double>& q_values) const;

    double compute_reward(const AcupunctureState& state,
                          const AcupunctureAction& action,
                          const AcupunctureState& next_state) const;

    void init_default_actions();

    std::string discretize_deqi(double deqi) const;
    std::string discretize_pain(double pain) const;
    std::string discretize_duration(double duration) const;
};

} // namespace tcm

#include "test_framework.h"
#include "q_learning_advisor.h"
#include <cmath>
#include <random>
#include <algorithm>
#include <map>

using namespace tcm;
using namespace tcm_test;

static AcupunctureState make_state(const std::string& vol, const std::string& mer,
                                   double deqi, double pain, double dur = 0) {
    AcupunctureState s;
    s.volunteer_id = vol;
    s.meridian_id = mer;
    s.deqi_intensity = deqi;
    s.pain_level = pain;
    s.current_duration_min = dur;
    s.session_count = 1;
    return s;
}

TEST_CAT(QLearn, 初始化探索率正确) {
    QLearningAdvisor q;
    q.initialize(0.001, 0.01, 0.95, 0.2);
    return approx_eq(q.get_exploration_rate(), 0.2, 1e-6);
}

TEST_CAT(QLearn, 推荐动作返回合法动作) {
    QLearningAdvisor q;
    q.initialize(0.001, 0.01, 0.95, 0.2);
    auto state = make_state("V001", "ST", 0.7, 0.5);
    auto res = q.recommend_action(state);
    auto acts = q.get_all_actions();
    bool valid_idx = res.recommended_action.action_index >= 0
                  && res.recommended_action.action_index < (int)acts.size();
    return valid_idx && res.confidence >= 0.0 && res.confidence <= 1.0
        && res.action_probabilities.size() == acts.size()
        && !std::isnan(res.state_value);
}

TEST_CAT(QLearn, 动作空间18种组合) {
    QLearningAdvisor q;
    q.initialize();
    auto acts = q.get_all_actions();
    return q.get_action_count() == 18 && (int)acts.size() == 18;
}

TEST_CAT(QLearn, 记录反馈后更新次数增加) {
    QLearningAdvisor q;
    q.initialize(0.001, 0.01, 0.95, 0.0);
    auto s1 = make_state("V001", "ST", 0.5, 0.5);
    auto r1 = q.recommend_action(s1);
    auto s2 = s1;
    s2.deqi_intensity = 0.8;
    s2.pain_level = 0.2;
    s2.current_duration_min = r1.recommended_action.needle_retention_min;
    auto before = q.get_total_updates();
    q.record_result(s1, r1.recommended_action, 0.8, s2, true);
    auto after = q.get_total_updates();
    return after > before;
}

TEST_CAT(QLearn, ActorCritic无振荡收敛到最优策略) {
    QLearningAdvisor q;
    q.initialize(0.001, 0.01, 0.95, 0.2);
    q.set_exploration_rate(0.2);
    const int BEST_ACTION = 3;
    const int EPISODES = 1000;
    auto state = make_state("V_CONV", "ST", 0.6, 0.6);
    std::mt19937 rng(777);
    std::uniform_real_distribution<> rd(0, 1);
    int last_100 = 0;
    std::vector<int> best_count_window;
    int window_best = 0;

    for (int ep = 0; ep < EPISODES; ++ep) {
        auto res = q.recommend_action(state);
        auto act = res.recommended_action;
        double reward;
        if (act.action_index == BEST_ACTION) {
            reward = 1.0 + rd(rng) * 0.2;
            if (ep >= EPISODES - 100) last_100++;
            window_best++;
        } else {
            reward = -0.2 + rd(rng) * 0.2;
        }
        auto ns = state;
        ns.current_duration_min = act.needle_retention_min;
        ns.deqi_intensity = 0.7 + rd(rng) * 0.2;
        ns.pain_level = 0.3 + rd(rng) * 0.2;
        q.record_result(state, act, reward, ns, true);
        q.decay_exploration();

        if ((ep + 1) % 100 == 0) {
            best_count_window.push_back(window_best);
            window_best = 0;
        }
    }

    int stable = 0;
    for (size_t i = 1; i < best_count_window.size(); ++i) {
        if (best_count_window[i] >= best_count_window[i - 1] * 0.7) {
            stable++;
        }
    }

    q.set_exploration_rate(0.0);
    int greedy_best = 0;
    for (int i = 0; i < 50; ++i) {
        auto r = q.recommend_action(state);
        if (r.recommended_action.action_index == BEST_ACTION) greedy_best++;
    }
    return (last_100 >= 50) && (stable >= (int)best_count_window.size() - 2) && (greedy_best >= 35);
}

TEST_CAT(QLearn, 探索率随衰减降低) {
    QLearningAdvisor q;
    q.initialize(0.001, 0.01, 0.95, 0.5);
    auto initial = q.get_exploration_rate();
    for (int i = 0; i < 50; ++i) q.decay_exploration();
    return q.get_exploration_rate() < initial;
}

TEST_CAT(QLearn, 状态数随学习增长) {
    QLearningAdvisor q;
    q.initialize();
    q.set_exploration_rate(0.0);
    auto before = q.get_state_count();
    for (int i = 0; i < 10; ++i) {
        auto s = make_state("V" + std::to_string(i), "ST", 0.5, 0.5);
        auto r = q.recommend_action(s);
        auto ns = s;
        ns.current_duration_min = r.recommended_action.needle_retention_min;
        q.record_result(s, r.recommended_action, 0.5, ns, true);
    }
    return q.get_state_count() >= before + 1;
}

TEST_CAT(QLearn, 重置后状态动作计数归零) {
    QLearningAdvisor q;
    q.initialize();
    auto s = make_state("V001", "ST", 0.5, 0.5);
    auto r = q.recommend_action(s);
    q.record_result(s, r.recommended_action, 0.5, s, true);
    q.reset_q_table();
    return q.get_state_count() == 0 && q.get_total_updates() == 0;
}

TEST_CAT(QLearn, 动作索引双向转换) {
    bool all_ok = true;
    for (int i = 0; i < 18; ++i) {
        auto a = QLearningAdvisor::action_from_index(i);
        auto j = QLearningAdvisor::action_to_index(a);
        if (i != j) all_ok = false;
    }
    return all_ok;
}

TEST_CAT(QLearn, ε贪婪_ε为零时动作一致) {
    QLearningAdvisor q;
    q.initialize(0.001, 0.01, 0.95, 0.0);
    auto s = make_state("V001", "ST", 0.5, 0.5);
    int idx0 = q.recommend_action(s).recommended_action.action_index;
    int same = 0;
    for (int i = 0; i < 20; ++i) {
        if (q.recommend_action(s).recommended_action.action_index == idx0) same++;
    }
    return same == 20;
}

TEST_CAT(QLearn, 奖励归一化防止大梯度振荡) {
    QLearningAdvisor q;
    q.initialize(0.001, 0.01, 0.95, 0.0);
    auto s = make_state("V_OSC", "SP", 0.5, 0.5);
    auto act = QLearningAdvisor::action_from_index(0);
    auto ns = s;
    ns.current_duration_min = act.needle_retention_min;
    q.record_result(s, act, 1000.0, ns, true);
    q.record_result(s, act, -1000.0, ns, true);
    auto res = q.recommend_action(s);
    bool finite = !std::isnan(res.state_value) && std::isfinite(res.state_value);
    for (auto p : res.action_probabilities) {
        if (!std::isfinite(p) || p < 0 || p > 1.01) finite = false;
    }
    return finite;
}

TEST_CAT(QLearn, TD误差计算有效) {
    QLearningAdvisor q;
    q.initialize(0.001, 0.01, 0.95, 0.0);
    auto s = make_state("V_TD", "SP", 0.5, 0.5);
    auto ns = s;
    ns.deqi_intensity = 0.8;
    double td = q.compute_td_error(1.0, "stateA", "stateB", false);
    return std::isfinite(td) && td >= -10 && td <= 10;
}

TEST_CAT(QLearn, 高负奖励使动作被避免) {
    QLearningAdvisor q;
    q.initialize(0.01, 0.1, 0.9, 0.0);
    auto s = make_state("V_PEN", "SP", 0.5, 0.5);
    int bad_action = 5;
    for (int i = 0; i < 200; ++i) {
        auto act = QLearningAdvisor::action_from_index(bad_action);
        auto ns = s;
        ns.current_duration_min = act.needle_retention_min;
        q.record_result(s, act, -1.0, ns, true);
    }
    q.set_exploration_rate(0.0);
    int pick_bad = 0;
    for (int i = 0; i < 30; ++i) {
        if (q.recommend_action(s).recommended_action.action_index == bad_action) pick_bad++;
    }
    return pick_bad <= 3;
}

TEST_CAT(QLearn, 异常空状态不崩溃) {
    QLearningAdvisor q;
    q.initialize();
    AcupunctureState s;
    s.volunteer_id = "";
    s.meridian_id = "";
    auto r = q.recommend_action(s);
    bool ok = r.confidence >= 0.0;
    q.record_result(s, r.recommended_action, 0.0, s, true);
    return ok;
}

TEST_CAT(QLearn, 平均奖励在合理范围) {
    QLearningAdvisor q;
    q.initialize();
    auto s = make_state("V001", "ST", 0.5, 0.5);
    for (int i = 0; i < 20; ++i) {
        auto r = q.recommend_action(s);
        q.record_result(s, r.recommended_action, 0.5, s, true);
    }
    double avg = q.get_average_reward();
    return avg >= -1.0 && avg <= 2.0;
}

TEST_CAT(QLearn, Top动作返回N个不超总数) {
    QLearningAdvisor q;
    q.initialize();
    auto s = make_state("V001", "ST", 0.5, 0.5);
    auto res = q.recommend_action(s);
    return (int)res.top_actions.size() <= q.get_action_count()
        && (int)res.action_values.size() <= q.get_action_count();
}

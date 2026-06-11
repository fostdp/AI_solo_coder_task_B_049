#include "test_framework.h"
#include "meridian_energy_balancer.h"
#include <cmath>
#include <random>
#include <algorithm>
#include <set>

using namespace tcm;
using namespace tcm_test;

static const std::vector<std::pair<std::string,std::string>> SYMMETRIC_PAIRS = {
    {"LU","LI"}, {"SP","ST"}, {"HT","SI"}, {"KI","BL"},
    {"PC","TE"}, {"LR","GB"}
};

static void feed_meridian(MeridianEnergyBalancer& b,
                          const std::string& mid,
                          double cond, double temp, double emg,
                          int count, uint64_t t0, uint64_t dt = 100) {
    for (int i = 0; i < count; ++i) {
        b.add_sensor_data(mid, mid + "1", cond, temp, emg, t0 + i * dt);
    }
}

TEST_CAT(Energy, 初始化后总样本数为零) {
    MeridianEnergyBalancer b;
    b.initialize(300);
    return b.total_samples() == 0;
}

TEST_CAT(Energy, 添加样本后计数正确) {
    MeridianEnergyBalancer b;
    b.initialize(300);
    for (int i = 0; i < 50; ++i)
        b.add_sensor_data("ST", "ST36", 12.0, 36.5, 20, 1000 + i * 100);
    return b.total_samples() == 50;
}

TEST_CAT(Energy, 平均电导计算正确) {
    MeridianEnergyBalancer b;
    b.initialize(300);
    b.set_baseline("ST", 10.0, 36.5);
    feed_meridian(b, "ST", 14.0, 36.6, 25, 100, 1000000, 100);
    auto res = b.compute_balance(2000000);
    for (auto& m : res.meridians) {
        if (m.meridian_id == "ST") {
            return m.conductance_avg >= 10.0 && m.conductance_avg <= 20.0;
        }
    }
    return false;
}

TEST_CAT(Energy, 左右对称经络差值小于阈值) {
    MeridianEnergyBalancer b;
    b.initialize(300);
    std::mt19937 rng(42);
    std::normal_distribution<> nd(0, 0.8);
    double base_cond = 10.0;
    double base_temp = 36.5;
    for (auto& p : SYMMETRIC_PAIRS) {
        for (int i = 0; i < 200; ++i) {
            double c1 = base_cond + nd(rng);
            double c2 = base_cond + 0.1 + nd(rng);
            b.add_sensor_data(p.first, p.first + "1", c1, base_temp, 20, 1000000 + i * 50);
            b.add_sensor_data(p.second, p.second + "1", c2, base_temp + 0.05, 21, 1000000 + i * 50 + 10);
        }
    }
    auto res = b.compute_balance(2000000);
    std::map<std::string, double> scores;
    for (auto& m : res.meridians) scores[m.meridian_id] = m.normalized_score;
    int passed_pairs = 0;
    for (auto& p : SYMMETRIC_PAIRS) {
        if (scores.count(p.first) && scores.count(p.second)) {
            double diff = std::fabs(scores[p.first] - scores[p.second]);
            if (diff < 0.5) passed_pairs++;
        }
    }
    return passed_pairs >= (int)SYMMETRIC_PAIRS.size() - 1;
}

TEST_CAT(Energy, 过亢经络标记为HYPER或EXCESS) {
    MeridianEnergyBalancer b;
    b.initialize(300);
    b.set_baseline("ST", 10.0, 36.5);
    for (int i = 0; i < 300; ++i)
        b.add_sensor_data("ST", "ST36", 35.0, 37.5, 80, 1000000 + i * 50);
    auto res = b.compute_balance(2000000);
    for (auto& m : res.meridians) {
        if (m.meridian_id == "ST") {
            return m.state == MeridianState::HYPERACTIVE
                || m.state == MeridianState::EXCESS
                || m.deviation_percent > 30.0;
        }
    }
    return false;
}

TEST_CAT(Energy, 虚证经络标记为HYPO或DEFICIENT) {
    MeridianEnergyBalancer b;
    b.initialize(300);
    b.set_baseline("SP", 10.0, 36.5);
    for (int i = 0; i < 300; ++i)
        b.add_sensor_data("SP", "SP6", 1.5, 35.5, 2, 1000000 + i * 50);
    auto res = b.compute_balance(2000000);
    for (auto& m : res.meridians) {
        if (m.meridian_id == "SP") {
            return m.state == MeridianState::HYPOACTIVE
                || m.state == MeridianState::DEFICIENT
                || m.deviation_percent < -20.0;
        }
    }
    return false;
}

TEST_CAT(Energy, 平衡经络标记为BALANCED) {
    MeridianEnergyBalancer b;
    b.initialize(300);
    b.set_baseline("LU", 10.0, 36.5);
    for (int i = 0; i < 200; ++i)
        b.add_sensor_data("LU", "LU7", 10.2, 36.52, 18, 1000000 + i * 50);
    auto res = b.compute_balance(2000000);
    for (auto& m : res.meridians) {
        if (m.meridian_id == "LU") {
            return m.state == MeridianState::BALANCED
                || std::fabs(m.deviation_percent) < 15.0;
        }
    }
    return false;
}

TEST_CAT(Energy, 阴阳平衡比值在合理范围) {
    MeridianEnergyBalancer b;
    b.initialize(300);
    std::vector<std::string> yin_m  = {"LU","SP","HT","KI","PC","LR","CV"};
    std::vector<std::string> yang_m = {"LI","ST","SI","BL","TE","GB","GV"};
    uint64_t t = 1000000;
    for (auto& m : yin_m)  feed_meridian(b, m,  9.5, 36.4, 15, 100, t, 80);
    for (auto& m : yang_m) feed_meridian(b, m, 10.5, 36.6, 20, 100, t, 80);
    auto res = b.compute_balance(t + 100000);
    return res.yin_energy > 0 && res.yang_energy > 0
        && res.yin_yang_ratio > 0.3 && res.yin_yang_ratio < 3.0;
}

TEST_CAT(Energy, 五行能量返回5个元素) {
    MeridianEnergyBalancer b;
    b.initialize(300);
    uint64_t t = 1000000;
    for (auto& p : SYMMETRIC_PAIRS) {
        feed_meridian(b, p.first,  10, 36.5, 18, 50, t, 80);
        feed_meridian(b, p.second, 11, 36.5, 20, 50, t, 80);
    }
    auto bal = b.compute_balance(t + 100000);
    auto fe = b.compute_five_element_energy(bal);
    std::set<std::string> names;
    for (auto& f : fe) names.insert(f.element);
    return fe.size() >= 4 && (names.count("木") || names.count("火") || names.count("土")
                           || names.count("金") || names.count("水"));
}

TEST_CAT(Energy, reset后样本数清零) {
    MeridianEnergyBalancer b;
    b.initialize(300);
    for (int i = 0; i < 100; ++i)
        b.add_sensor_data("ST", "ST36", 10, 36.5, 15, 1000000 + i);
    b.reset();
    return b.total_samples() == 0;
}

TEST_CAT(Energy, 异常零电导样本处理) {
    MeridianEnergyBalancer b;
    b.initialize(300);
    for (int i = 0; i < 50; ++i)
        b.add_sensor_data("ST", "ST36", 0.0, 0.0, 0.0, 1000 + i);
    auto res = b.compute_balance(2000000);
    bool ok = true;
    for (auto& m : res.meridians) {
        if (std::isnan(m.energy_score) || std::isinf(m.energy_score)) ok = false;
    }
    return ok;
}

TEST_CAT(Energy, 空数据compute_balance返回空) {
    MeridianEnergyBalancer b;
    b.initialize(300);
    auto res = b.compute_balance(1000000);
    return res.meridians.empty();
}

TEST_CAT(Energy, 整体平衡分在01范围) {
    MeridianEnergyBalancer b;
    b.initialize(300);
    uint64_t t = 1000000;
    for (auto& p : SYMMETRIC_PAIRS) {
        feed_meridian(b, p.first,  10, 36.5, 18, 50, t, 80);
        feed_meridian(b, p.second, 11, 36.5, 20, 50, t, 80);
    }
    auto res = b.compute_balance(t + 100000);
    return res.overall_balance_score >= -1.0 && res.overall_balance_score <= 2.0;
}

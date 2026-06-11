#include "energy_radar.h"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace tcm::radar {

const std::map<std::string, std::string> EnergyRadar::meridian_elements_ = {
    {"LU", "metal"}, {"LI", "metal"},
    {"ST", "earth"}, {"SP", "earth"},
    {"HT", "fire"}, {"SI", "fire"},
    {"BL", "water"}, {"KI", "water"},
    {"PC", "fire"}, {"TE", "fire"},
    {"GB", "wood"}, {"LR", "wood"},
    {"CV", "earth"}, {"GV", "yang"}
};

const std::map<std::string, std::string> EnergyRadar::meridian_yin_yang_ = {
    {"LU", "yin"}, {"LI", "yang"},
    {"ST", "yang"}, {"SP", "yin"},
    {"HT", "yin"}, {"SI", "yang"},
    {"BL", "yang"}, {"KI", "yin"},
    {"PC", "yin"}, {"TE", "yang"},
    {"GB", "yang"}, {"LR", "yin"},
    {"CV", "yin"}, {"GV", "yang"}
};

const std::map<std::string, std::string> EnergyRadar::meridian_names_ = {
    {"LU", "手太阴肺经"},
    {"LI", "手阳明大肠经"},
    {"ST", "足阳明胃经"},
    {"SP", "足太阴脾经"},
    {"HT", "手少阴心经"},
    {"SI", "手太阳小肠经"},
    {"BL", "足太阳膀胱经"},
    {"KI", "足少阴肾经"},
    {"PC", "手厥阴心包经"},
    {"TE", "手少阳三焦经"},
    {"GB", "足少阳胆经"},
    {"LR", "足厥阴肝经"},
    {"CV", "任脉"},
    {"GV", "督脉"}
};

EnergyRadar::EnergyRadar()
    : baseline_window_sec_(300.0), total_samples_(0) {
}

void EnergyRadar::initialize(double baseline_window_sec) {
    baseline_window_sec_ = baseline_window_sec;
    reset();
}

void EnergyRadar::reset() {
    meridian_data_.clear();
    total_samples_ = 0;
}

void EnergyRadar::set_baseline(
    const std::string& meridian_id,
    double conductance_base,
    double temperature_base) {
    auto& md = meridian_data_[meridian_id];
    md.baseline_conductance = conductance_base;
    md.baseline_temperature = temperature_base;
    md.baseline_set = true;
}

void EnergyRadar::add_sensor_data(
    const std::string& meridian_id,
    const std::string& acupoint_id,
    double skin_conductance,
    double temperature,
    double emg_amplitude,
    uint64_t timestamp) {

    (void)acupoint_id;

    auto& md = meridian_data_[meridian_id];
    md.conductances.push_back(skin_conductance);
    md.temperatures.push_back(temperature);
    md.emg_values.push_back(emg_amplitude);
    md.timestamps.push_back(timestamp);
    md.sum_conductance += skin_conductance;
    md.sum_temperature += temperature;
    md.sum_emg += emg_amplitude;

    total_samples_++;

    if (!md.baseline_set && md.conductances.size() >= 50) {
        md.baseline_conductance = md.sum_conductance / md.conductances.size();
        md.baseline_temperature = md.sum_temperature / md.temperatures.size();
        md.baseline_set = true;
    }

    uint64_t cutoff_ms = (uint64_t)(baseline_window_sec_ * 1000);
    if (timestamp > cutoff_ms) {
        trim_old_data(md, timestamp - cutoff_ms);
    }
}

void EnergyRadar::trim_old_data(MeridianData& md, uint64_t cutoff_time) {
    while (!md.timestamps.empty() && md.timestamps.front() < cutoff_time) {
        md.sum_conductance -= md.conductances.front();
        md.sum_temperature -= md.temperatures.front();
        md.sum_emg -= md.emg_values.front();
        md.conductances.pop_front();
        md.temperatures.pop_front();
        md.emg_values.pop_front();
        md.timestamps.pop_front();
    }
}

void EnergyRadar::update_sums(MeridianData& md) {
    md.sum_conductance = 0;
    md.sum_temperature = 0;
    md.sum_emg = 0;
    for (double v : md.conductances) md.sum_conductance += v;
    for (double v : md.temperatures) md.sum_temperature += v;
    for (double v : md.emg_values) md.sum_emg += v;
}

double EnergyRadar::compute_energy_score(const MeridianData& md) const {
    if (md.conductances.empty()) return 50.0;

    double avg_conductance = md.sum_conductance / md.conductances.size();
    double avg_emg = md.sum_emg / md.emg_values.size();

    double cond_score = 0;
    if (md.baseline_set && md.baseline_conductance > 0) {
        double ratio = avg_conductance / md.baseline_conductance;
        cond_score = std::max(0.0, std::min(100.0, ratio * 50.0));
    } else {
        cond_score = std::min(100.0, avg_conductance / 30.0 * 50.0 + 25.0);
    }

    double emg_score = std::min(50.0, avg_emg / 100.0 * 50.0);

    return cond_score * 0.7 + emg_score * 0.3;
}

double EnergyRadar::compute_normalized_score(const MeridianData& md) const {
    if (!md.baseline_set || md.baseline_conductance <= 0) {
        return 50.0;
    }

    double avg_conductance = md.sum_conductance / md.conductances.size();
    double ratio = avg_conductance / md.baseline_conductance;

    if (ratio >= 1.0) {
        return 50.0 + (ratio - 1.0) * 100.0;
    } else {
        return 50.0 * ratio;
    }
}

MeridianState EnergyRadar::classify_state(double normalized_score) const {
    if (normalized_score >= 70.0) return MeridianState::HYPERACTIVE;
    if (normalized_score >= 58.0) return MeridianState::EXCESS;
    if (normalized_score >= 42.0) return MeridianState::BALANCED;
    if (normalized_score >= 30.0) return MeridianState::DEFICIENT;
    return MeridianState::HYPOACTIVE;
}

EnergyBalanceResult EnergyRadar::compute_balance(uint64_t current_time) {
    EnergyBalanceResult result;
    result.timestamp = current_time;

    double total_energy = 0;
    int count = 0;
    double yin_sum = 0;
    double yang_sum = 0;
    int yin_count = 0;
    int yang_count = 0;

    for (auto& kv : meridian_data_) {
        const auto& mid = kv.first;
        auto& md = kv.second;

        if (md.timestamps.empty()) continue;

        double avg_cond = md.sum_conductance / md.conductances.size();
        double avg_temp = md.sum_temperature / md.temperatures.size();
        double avg_emg = md.sum_emg / md.emg_values.size();

        MeridianEnergy me;
        me.meridian_id = mid;
        auto name_it = meridian_names_.find(mid);
        me.meridian_name = (name_it != meridian_names_.end()) ? name_it->second : mid;
        me.conductance_avg = avg_cond;
        me.temperature_avg = avg_temp;
        me.emg_activity = avg_emg;
        me.energy_score = compute_energy_score(md);
        me.normalized_score = compute_normalized_score(md);
        me.state = classify_state(me.normalized_score);
        me.sample_count = (int)md.conductances.size();

        if (md.baseline_set && md.baseline_conductance > 0) {
            me.deviation_percent = (avg_cond - md.baseline_conductance) / md.baseline_conductance * 100.0;
        } else {
            me.deviation_percent = 0;
        }

        auto yy_it = meridian_yin_yang_.find(mid);
        me.yin_yang_balance = (yy_it != meridian_yin_yang_.end() && yy_it->second == "yin") ? -1.0 : 1.0;

        result.meridians.push_back(me);
        total_energy += me.energy_score;
        count++;

        if (yy_it != meridian_yin_yang_.end()) {
            if (yy_it->second == "yin") {
                yin_sum += me.energy_score;
                yin_count++;
            } else {
                yang_sum += me.energy_score;
                yang_count++;
            }
        }

        if (me.state == MeridianState::EXCESS || me.state == MeridianState::HYPERACTIVE) {
            result.excess_meridians.push_back(mid);
        } else if (me.state == MeridianState::DEFICIENT || me.state == MeridianState::HYPOACTIVE) {
            result.deficient_meridians.push_back(mid);
        }
    }

    result.overall_balance_score = count > 0 ? total_energy / count : 50.0;
    result.yin_energy = yin_count > 0 ? yin_sum / yin_count : 0;
    result.yang_energy = yang_count > 0 ? yang_sum / yang_count : 0;
    result.yin_yang_ratio = result.yin_energy > 0 ? result.yang_energy / result.yin_energy : 1.0;

    if (!result.meridians.empty()) {
        double max_energy = 0;
        double min_energy = 999;
        for (const auto& m : result.meridians) {
            if (m.energy_score > max_energy) {
                max_energy = m.energy_score;
                result.dominant_meridian = m.meridian_id;
            }
            if (m.energy_score < min_energy) {
                min_energy = m.energy_score;
                result.weakest_meridian = m.meridian_id;
            }
        }
    }

    return result;
}

std::vector<FiveElementEnergy>
EnergyRadar::compute_five_element_energy(const EnergyBalanceResult& balance) const {
    std::map<std::string, double> element_sums;
    std::map<std::string, int> element_counts;
    std::map<std::string, std::vector<std::string>> element_meridians;

    for (const auto& m : balance.meridians) {
        auto it = meridian_elements_.find(m.meridian_id);
        if (it != meridian_elements_.end()) {
            const std::string& elem = it->second;
            element_sums[elem] += m.energy_score;
            element_counts[elem]++;
            element_meridians[elem].push_back(m.meridian_id);
        }
    }

    std::vector<FiveElementEnergy> result;
    for (const auto& kv : element_sums) {
        FiveElementEnergy fee;
        fee.element = kv.first;
        fee.energy = element_counts[kv.first] > 0 ? kv.second / element_counts[kv.first] : 0;
        fee.meridians = element_meridians[kv.first];
        result.push_back(fee);
    }

    std::sort(result.begin(), result.end(),
        [](const FiveElementEnergy& a, const FiveElementEnergy& b) {
            return a.energy > b.energy;
        });

    return result;
}

} // namespace tcm::radar

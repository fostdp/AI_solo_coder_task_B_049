#pragma once
#include <string>
#include <vector>
#include <map>
#include <deque>
#include <cstdint>

namespace tcm {

enum class MeridianState {
    BALANCED = 0,
    EXCESS = 1,
    DEFICIENT = 2,
    HYPERACTIVE = 3,
    HYPOACTIVE = 4
};

struct MeridianEnergy {
    std::string meridian_id;
    std::string meridian_name;
    double conductance_avg;
    double temperature_avg;
    double emg_activity;
    double energy_score;
    double normalized_score;
    MeridianState state;
    double deviation_percent;
    double yin_yang_balance;
    int sample_count;
};

struct EnergyBalanceResult {
    std::vector<MeridianEnergy> meridians;
    double overall_balance_score;
    double yin_energy;
    double yang_energy;
    double yin_yang_ratio;
    std::string dominant_meridian;
    std::string weakest_meridian;
    std::vector<std::string> excess_meridians;
    std::vector<std::string> deficient_meridians;
    uint64_t timestamp;
};

struct FiveElementEnergy {
    std::string element;
    double energy;
    std::vector<std::string> meridians;
};

class MeridianEnergyBalancer {
public:
    MeridianEnergyBalancer();
    ~MeridianEnergyBalancer() = default;

    void initialize(double baseline_window_sec = 300.0);

    void add_sensor_data(const std::string& meridian_id,
                         const std::string& acupoint_id,
                         double skin_conductance,
                         double temperature,
                         double emg_amplitude,
                         uint64_t timestamp);

    EnergyBalanceResult compute_balance(uint64_t current_time);

    std::vector<FiveElementEnergy> compute_five_element_energy(
        const EnergyBalanceResult& balance) const;

    void set_baseline(const std::string& meridian_id,
                      double conductance_base,
                      double temperature_base);

    void reset();

    size_t total_samples() const { return total_samples_; }

private:
    struct MeridianData {
        std::deque<double> conductances;
        std::deque<double> temperatures;
        std::deque<double> emg_values;
        std::deque<uint64_t> timestamps;
        double baseline_conductance;
        double baseline_temperature;
        bool baseline_set;
        double sum_conductance;
        double sum_temperature;
        double sum_emg;
    };

    std::map<std::string, MeridianData> meridian_data_;
    double baseline_window_sec_;
    size_t total_samples_;

    static const std::map<std::string, std::string> meridian_elements_;
    static const std::map<std::string, std::string> meridian_yin_yang_;
    static const std::map<std::string, std::string> meridian_names_;

    double compute_energy_score(const MeridianData& md) const;
    double compute_normalized_score(const MeridianData& md) const;
    MeridianState classify_state(double normalized_score) const;

    void trim_old_data(MeridianData& md, uint64_t cutoff_time);
    void update_sums(MeridianData& md);
};

} // namespace tcm

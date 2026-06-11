#pragma once
#include <string>
#include <vector>
#include <map>
#include <deque>

namespace tcm {

enum class NeedleTechnique {
    RESTING = 0,
    LIFTING_THRUSTING = 1,
    TWIRLING = 2,
    REINFORCING = 3,
    REDUCING = 4,
    EVEN_METHOD = 5
};

struct EMGFeature {
    double rms;
    double mav;
    double zero_crossing_rate;
    double peak_frequency;
    double mean_frequency;
    double median_frequency;
    double spectral_entropy;
    double waveform_length;
    double variance;
    double kurtosis;
    double skewness;
    double hjorth_activity;
    double hjorth_mobility;
    double hjorth_complexity;
    double band_power_low;
    double band_power_mid;
    double band_power_high;
};

struct TechniqueAnalysisResult {
    NeedleTechnique technique;
    std::string technique_name;
    double confidence;
    EMGFeature features;
    double lifting_frequency_hz;
    double twirling_frequency_hz;
    double amplitude_ratio;
    bool is_active;
    uint64_t window_start;
    uint64_t window_end;
};

struct Conv1DLayer {
    int in_channels;
    int out_channels;
    int kernel_size;
    int stride;
    int padding;
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;
};

struct DenseLayer {
    int in_features;
    int out_features;
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;
    std::string activation;
};

class NeedleTechniqueClassifier {
public:
    NeedleTechniqueClassifier();
    ~NeedleTechniqueClassifier() = default;

    void initialize(int sample_rate = 1000, int window_size = 500);

    TechniqueAnalysisResult analyze(const std::vector<double>& emg_signal, uint64_t timestamp);

    void add_sample(const std::vector<double>& emg_signal, NeedleTechnique label);

    bool train(const std::vector<std::vector<double>>& samples,
               const std::vector<NeedleTechnique>& labels,
               int epochs = 50, double learning_rate = 0.01);

    std::vector<std::vector<double>> augment_samples(
        const std::vector<std::vector<double>>& samples,
        const std::vector<NeedleTechnique>& labels,
        std::vector<NeedleTechnique>& augmented_labels,
        int augment_factor = 4) const;

    static std::string technique_to_string(NeedleTechnique t);
    static NeedleTechnique string_to_technique(const std::string& s);

    void reset_buffer();
    size_t buffer_size() const { return emg_buffer_.size(); }

    static std::vector<double> denoise_signal(const std::vector<double>& signal, int sample_rate);

private:
    int sample_rate_;
    int window_size_;
    std::deque<double> emg_buffer_;
    std::deque<uint64_t> timestamp_buffer_;
    std::mt19937 rng_;

    std::vector<std::vector<double>> training_samples_;
    std::vector<NeedleTechnique> training_labels_;

    std::vector<Conv1DLayer> conv_layers_;
    std::vector<DenseLayer> dense_layers_;
    std::vector<std::string> class_names_;

    EMGFeature extract_features(const std::vector<double>& signal) const;

    double compute_rms(const std::vector<double>& signal) const;
    double compute_mav(const std::vector<double>& signal) const;
    double compute_zero_crossing_rate(const std::vector<double>& signal) const;
    double compute_waveform_length(const std::vector<double>& signal) const;
    double compute_kurtosis(const std::vector<double>& signal, double mean, double std) const;
    double compute_skewness(const std::vector<double>& signal, double mean, double std) const;

    void compute_fft(const std::vector<double>& signal,
                     std::vector<double>& magnitudes,
                     std::vector<double>& frequencies) const;
    double compute_peak_frequency(const std::vector<double>& magnitudes,
                                  const std::vector<double>& freqs) const;
    double compute_mean_frequency(const std::vector<double>& magnitudes,
                                  const std::vector<double>& freqs) const;
    double compute_spectral_entropy(const std::vector<double>& magnitudes) const;

    void compute_hjorth(const std::vector<double>& signal,
                        double& activity, double& mobility, double& complexity) const;

    std::vector<double> conv1d_forward(const std::vector<double>& input,
                                       const Conv1DLayer& layer) const;
    std::vector<double> relu(const std::vector<double>& x) const;
    std::vector<double> maxpool1d(const std::vector<double>& x, int pool_size) const;
    std::vector<double> dense_forward(const std::vector<double>& input,
                                      const DenseLayer& layer) const;
    std::vector<double> softmax(const std::vector<double>& x) const;

    std::vector<double> flatten_features(const EMGFeature& f) const;
    NeedleTechnique predict_class(const std::vector<double>& features, double& confidence) const;

    void build_default_model();
    void init_random_weights();

    std::vector<double> denoise_signal_internal(const std::vector<double>& signal) const;
    std::vector<double> moving_average_filter(const std::vector<double>& signal, int window) const;
    std::vector<double> remove_baseline_drift(const std::vector<double>& signal) const;
    std::vector<double> notch_filter_50hz(const std::vector<double>& signal) const;
    std::vector<double> time_warp(const std::vector<double>& signal, double sigma) const;
    std::vector<double> time_shift(const std::vector<double>& signal, int shift) const;
    std::vector<double> amplitude_scale(const std::vector<double>& signal, double scale) const;
    std::vector<double> add_gaussian_noise(const std::vector<double>& signal, double snr_db) const;
    double compute_signal_power(const std::vector<double>& signal) const;
};

} // namespace tcm

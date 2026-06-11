#include "maneuver_cnn.h"
#include "../../common/include/thread_pool.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <complex>
#include <stdexcept>
#include <thread>

namespace tcm {
namespace cnn {

struct ManeuverCNN::Impl {
    common::ThreadPool pool;
};

ManeuverCNN::ManeuverCNN()
    : sample_rate_(1000), window_size_(500), num_threads_(0), rng_(42) {
    build_default_model();
}

void ManeuverCNN::initialize(int sample_rate, int window_size, int num_threads) {
    sample_rate_ = sample_rate;
    window_size_ = window_size;
    num_threads_ = num_threads > 0 ? num_threads
               : (int)std::max(1u, std::thread::hardware_concurrency());
    emg_buffer_.clear();
    timestamp_buffer_.clear();
    rng_.seed(42);
    impl_ = std::make_unique<Impl>();
    impl_->pool = common::ThreadPool(num_threads_);
}

std::future<TechniqueAnalysisResult> ManeuverCNN::analyze_async(
    const std::vector<double>& emg_signal, uint64_t timestamp) {
    return impl_->pool.submit(&ManeuverCNN::analyze, this, emg_signal, timestamp);
}

void ManeuverCNN::build_default_model() {
    class_names_ = {
        "resting", "lifting_thrusting", "twirling",
        "reinforcing", "reducing", "even_method"
    };

    Conv1DLayer conv1;
    conv1.in_channels = 1;
    conv1.out_channels = 16;
    conv1.kernel_size = 5;
    conv1.stride = 1;
    conv1.padding = 2;
    conv_layers_.push_back(conv1);

    Conv1DLayer conv2;
    conv2.in_channels = 16;
    conv2.out_channels = 32;
    conv2.kernel_size = 3;
    conv2.stride = 1;
    conv2.padding = 1;
    conv_layers_.push_back(conv2);

    DenseLayer dense1;
    dense1.in_features = 17 + 16;
    dense1.out_features = 32;
    dense1.activation = "relu";
    dense_layers_.push_back(dense1);

    DenseLayer dense2;
    dense2.in_features = 32;
    dense2.out_features = 6;
    dense2.activation = "softmax";
    dense_layers_.push_back(dense2);

    init_random_weights();
}

void ManeuverCNN::init_random_weights() {
    std::mt19937 gen(42);
    std::normal_distribution<> d(0.0, 0.1);

    for (auto& layer : conv_layers_) {
        layer.weights.clear();
        layer.biases.assign(layer.out_channels, 0.0);
        for (int oc = 0; oc < layer.out_channels; ++oc) {
            std::vector<double> w;
            for (int ic = 0; ic < layer.in_channels; ++ic) {
                for (int k = 0; k < layer.kernel_size; ++k) {
                    w.push_back(d(gen));
                }
            }
            layer.weights.push_back(w);
        }
    }

    for (auto& layer : dense_layers_) {
        layer.weights.clear();
        layer.biases.assign(layer.out_features, 0.0);
        double scale = std::sqrt(2.0 / layer.in_features);
        for (int i = 0; i < layer.out_features; ++i) {
            std::vector<double> w;
            for (int j = 0; j < layer.in_features; ++j) {
                w.push_back(d(gen) * scale);
            }
            layer.weights.push_back(w);
        }
    }
}

TechniqueAnalysisResult
ManeuverCNN::analyze(const std::vector<double>& emg_signal, uint64_t timestamp) {
    TechniqueAnalysisResult result;
    result.window_start = timestamp;
    result.window_end = timestamp + emg_signal.size() * 1000 / sample_rate_;

    if (emg_signal.size() < 10) {
        result.technique = NeedleTechnique::RESTING;
        result.technique_name = "resting";
        result.confidence = 0.5;
        result.is_active = false;
        return result;
    }

    auto denoised = denoise_signal_internal(emg_signal);
    result.features = extract_features(denoised);

    result.lifting_frequency_hz = 0;
    result.twirling_frequency_hz = 0;
    result.amplitude_ratio = result.features.rms > 0 ?
        result.features.band_power_high / (result.features.band_power_low + 0.001) : 0;

    double rms_threshold = 5.0;
    result.is_active = result.features.rms > rms_threshold;

    auto feat_vec = flatten_features(result.features);
    double confidence = 0.0;
    auto tech = predict_class(feat_vec, confidence);
    result.technique = tech;
    result.technique_name = technique_to_string(tech);
    result.confidence = confidence;

    if (!result.is_active) {
        result.technique = NeedleTechnique::RESTING;
        result.technique_name = "resting";
    }

    return result;
}

EMGFeature ManeuverCNN::extract_features(const std::vector<double>& signal) const {
    EMGFeature f;
    int n = (int)signal.size();

    double sum = std::accumulate(signal.begin(), signal.end(), 0.0);
    double mean = sum / n;

    f.mav = compute_mav(signal);
    f.rms = compute_rms(signal);
    f.zero_crossing_rate = compute_zero_crossing_rate(signal);
    f.waveform_length = compute_waveform_length(signal);

    double var = 0.0;
    for (double x : signal) var += (x - mean) * (x - mean);
    f.variance = var / n;
    double std = std::sqrt(f.variance);

    f.kurtosis = compute_kurtosis(signal, mean, std);
    f.skewness = compute_skewness(signal, mean, std);

    compute_hjorth(signal, f.hjorth_activity, f.hjorth_mobility, f.hjorth_complexity);

    std::vector<double> magnitudes, freqs;
    compute_fft(signal, magnitudes, freqs);
    f.peak_frequency = compute_peak_frequency(magnitudes, freqs);
    f.mean_frequency = compute_mean_frequency(magnitudes, freqs);
    f.median_frequency = f.mean_frequency * 0.75;
    f.spectral_entropy = compute_spectral_entropy(magnitudes);

    f.band_power_low = 0;
    f.band_power_mid = 0;
    f.band_power_high = 0;

    for (size_t i = 0; i < freqs.size(); ++i) {
        double freq = freqs[i];
        double mag = magnitudes[i] * magnitudes[i];
        if (freq < 20) f.band_power_low += mag;
        else if (freq < 80) f.band_power_mid += mag;
        else if (freq < 200) f.band_power_high += mag;
    }

    return f;
}

double ManeuverCNN::compute_rms(const std::vector<double>& signal) const {
    double sum_sq = 0;
    for (double x : signal) sum_sq += x * x;
    return std::sqrt(sum_sq / signal.size());
}

double ManeuverCNN::compute_mav(const std::vector<double>& signal) const {
    double sum = 0;
    for (double x : signal) sum += std::abs(x);
    return sum / signal.size();
}

double ManeuverCNN::compute_zero_crossing_rate(const std::vector<double>& signal) const {
    int crossings = 0;
    for (size_t i = 1; i < signal.size(); ++i) {
        if ((signal[i-1] >= 0 && signal[i] < 0) ||
            (signal[i-1] < 0 && signal[i] >= 0)) {
            crossings++;
        }
    }
    return (double)crossings / (signal.size() / (double)sample_rate_);
}

double ManeuverCNN::compute_waveform_length(const std::vector<double>& signal) const {
    double length = 0;
    for (size_t i = 1; i < signal.size(); ++i) {
        length += std::abs(signal[i] - signal[i-1]);
    }
    return length;
}

double ManeuverCNN::compute_kurtosis(
    const std::vector<double>& signal, double mean, double std) const {
    if (std < 1e-8) return 0;
    double sum4 = 0;
    for (double x : signal) {
        double z = (x - mean) / std;
        sum4 += z * z * z * z;
    }
    return sum4 / signal.size() - 3.0;
}

double ManeuverCNN::compute_skewness(
    const std::vector<double>& signal, double mean, double std) const {
    if (std < 1e-8) return 0;
    double sum3 = 0;
    for (double x : signal) {
        double z = (x - mean) / std;
        sum3 += z * z * z;
    }
    return sum3 / signal.size();
}

void ManeuverCNN::compute_fft(
    const std::vector<double>& signal,
    std::vector<double>& magnitudes,
    std::vector<double>& frequencies) const {

    int n = (int)signal.size();
    int nfft = 1;
    while (nfft < n) nfft <<= 1;

    std::vector<std::complex<double>> x(nfft, 0);
    for (int i = 0; i < n; ++i) {
        x[i] = std::complex<double>(signal[i], 0);
    }

    for (int len = 2; len <= nfft; len <<= 1) {
        double ang = -2 * M_PI / len;
        std::complex<double> wlen(cos(ang), sin(ang));
        for (int i = 0; i < nfft; i += len) {
            std::complex<double> w(1, 0);
            for (int j = 0; j < len / 2; ++j) {
                auto u = x[i + j];
                auto v = x[i + j + len/2] * w;
                x[i + j] = u + v;
                x[i + j + len/2] = u - v;
                w *= wlen;
            }
        }
    }

    magnitudes.clear();
    frequencies.clear();
    int half = nfft / 2 + 1;
    for (int i = 0; i < half; ++i) {
        magnitudes.push_back(std::abs(x[i]) / n);
        frequencies.push_back((double)i * sample_rate_ / nfft);
    }
}

double ManeuverCNN::compute_peak_frequency(
    const std::vector<double>& magnitudes,
    const std::vector<double>& freqs) const {
    if (magnitudes.empty()) return 0;
    size_t max_idx = 0;
    double max_val = 0;
    for (size_t i = 1; i < magnitudes.size(); ++i) {
        if (magnitudes[i] > max_val) {
            max_val = magnitudes[i];
            max_idx = i;
        }
    }
    return freqs[max_idx];
}

double ManeuverCNN::compute_mean_frequency(
    const std::vector<double>& magnitudes,
    const std::vector<double>& freqs) const {
    double sum_mag = 0;
    double sum_freq_mag = 0;
    for (size_t i = 0; i < magnitudes.size(); ++i) {
        sum_mag += magnitudes[i];
        sum_freq_mag += freqs[i] * magnitudes[i];
    }
    return sum_mag > 0 ? sum_freq_mag / sum_mag : 0;
}

double ManeuverCNN::compute_spectral_entropy(
    const std::vector<double>& magnitudes) const {
    double total = 0;
    for (double m : magnitudes) total += m * m;
    if (total < 1e-10) return 0;

    double entropy = 0;
    for (double m : magnitudes) {
        double p = (m * m) / total;
        if (p > 1e-10) {
            entropy -= p * log(p);
        }
    }
    return entropy;
}

void ManeuverCNN::compute_hjorth(
    const std::vector<double>& signal,
    double& activity, double& mobility, double& complexity) const {

    int n = (int)signal.size();
    double mean = 0;
    for (double x : signal) mean += x;
    mean /= n;

    activity = 0;
    for (double x : signal) activity += (x - mean) * (x - mean);
    activity /= n;

    std::vector<double> diff1;
    for (int i = 1; i < n; ++i) {
        diff1.push_back(signal[i] - signal[i-1]);
    }
    double m1 = 0;
    for (double x : diff1) m1 += x;
    m1 /= diff1.size();
    double var1 = 0;
    for (double x : diff1) var1 += (x - m1) * (x - m1);
    var1 /= diff1.size();

    mobility = activity > 0 ? std::sqrt(var1 / activity) : 0;

    std::vector<double> diff2;
    for (size_t i = 1; i < diff1.size(); ++i) {
        diff2.push_back(diff1[i] - diff1[i-1]);
    }
    double m2 = 0;
    for (double x : diff2) m2 += x;
    m2 /= diff2.size();
    double var2 = 0;
    for (double x : diff2) var2 += (x - m2) * (x - m2);
    var2 /= diff2.size();

    double mobility2 = var1 > 0 ? std::sqrt(var2 / var1) : 0;
    complexity = mobility > 0 ? mobility2 / mobility : 0;
}

std::vector<double> ManeuverCNN::flatten_features(const EMGFeature& f) const {
    return {
        f.rms / 100.0,
        f.mav / 50.0,
        f.zero_crossing_rate / 200.0,
        f.peak_frequency / 100.0,
        f.mean_frequency / 100.0,
        f.median_frequency / 100.0,
        f.spectral_entropy / 5.0,
        f.waveform_length / 10000.0,
        f.variance / 10000.0,
        f.kurtosis / 10.0,
        f.skewness / 5.0,
        f.hjorth_activity / 10000.0,
        f.hjorth_mobility,
        f.hjorth_complexity / 5.0,
        f.band_power_low / 1000.0,
        f.band_power_mid / 1000.0,
        f.band_power_high / 1000.0
    };
}

NeedleTechnique ManeuverCNN::predict_class(
    const std::vector<double>& features, double& confidence) const {

    auto x = features;
    for (const auto& layer : dense_layers_) {
        x = dense_forward(x, layer);
    }

    int max_idx = 0;
    double max_val = x[0];
    for (size_t i = 1; i < x.size(); ++i) {
        if (x[i] > max_val) {
            max_val = x[i];
            max_idx = (int)i;
        }
    }

    confidence = max_val;
    return (NeedleTechnique)max_idx;
}

std::vector<double> ManeuverCNN::dense_forward(
    const std::vector<double>& input,
    const DenseLayer& layer) const {

    std::vector<double> output(layer.out_features, 0.0);

    for (int i = 0; i < layer.out_features; ++i) {
        double sum = layer.biases[i];
        for (int j = 0; j < layer.in_features && j < (int)input.size(); ++j) {
            if (j < (int)layer.weights[i].size()) {
                sum += input[j] * layer.weights[i][j];
            }
        }
        output[i] = sum;
    }

    if (layer.activation == "relu") {
        for (auto& v : output) v = std::max(0.0, v);
    } else if (layer.activation == "softmax") {
        double max_val = *std::max_element(output.begin(), output.end());
        double sum_exp = 0;
        for (auto& v : output) {
            v = std::exp(v - max_val);
            sum_exp += v;
        }
        if (sum_exp > 0) {
            for (auto& v : output) v /= sum_exp;
        }
    }

    return output;
}

std::vector<double> ManeuverCNN::relu(const std::vector<double>& x) const {
    std::vector<double> out(x.size());
    for (size_t i = 0; i < x.size(); ++i) out[i] = std::max(0.0, x[i]);
    return out;
}

void ManeuverCNN::add_sample(
    const std::vector<double>& emg_signal, NeedleTechnique label) {
    training_samples_.push_back(emg_signal);
    training_labels_.push_back(label);
}

bool ManeuverCNN::train(
    const std::vector<std::vector<double>>& samples,
    const std::vector<NeedleTechnique>& labels,
    int epochs, double learning_rate,
    int batch_size,
    double* loss_before, double* loss_after) {

    if (samples.size() != labels.size() || samples.empty()) return false;

    training_samples_ = samples;
    training_labels_ = labels;

    std::vector<NeedleTechnique> aug_labels;
    auto aug_samples = augment_samples(samples, labels, aug_labels, 4);

    init_random_weights();

    int n = (int)aug_samples.size();
    std::uniform_int_distribution<> dist(0, n - 1);

    double first_epoch_loss = 0;

    for (int ep = 0; ep < epochs; ++ep) {
        double loss_sum = 0;
        for (int b = 0; b < batch_size; ++b) {
            int idx = dist(rng_);
            const auto& sample = aug_samples[idx];
            auto label = aug_labels[idx];

            auto denoised = denoise_signal_internal(sample);
            auto feat = extract_features(denoised);
            auto feat_vec = flatten_features(feat);

            auto x = feat_vec;
            for (const auto& layer : dense_layers_) {
                x = dense_forward(x, layer);
            }

            std::vector<double> target(6, 0.0);
            target[(int)label] = 1.0;

            double loss = 0;
            for (size_t i = 0; i < x.size(); ++i) {
                loss += -target[i] * log(x[i] + 1e-10);
            }
            loss_sum += loss;

            std::vector<double> delta(x.size());
            for (size_t i = 0; i < x.size(); ++i) {
                delta[i] = x[i] - target[i];
            }

            for (int li = (int)dense_layers_.size() - 1; li >= 0; --li) {
                auto& layer = dense_layers_[li];
                std::vector<double> prev_delta(layer.in_features, 0.0);

                for (int i = 0; i < layer.out_features; ++i) {
                    double d = delta[i];
                    for (int j = 0; j < layer.in_features; ++j) {
                        if (j < (int)feat_vec.size() && j < (int)layer.weights[i].size()) {
                            double grad = d * feat_vec[j];
                            layer.weights[i][j] -= learning_rate * grad;
                            prev_delta[j] += d * layer.weights[i][j];
                        }
                    }
                    layer.biases[i] -= learning_rate * d;
                }

                if (layer.activation == "relu") {
                    for (int j = 0; j < layer.in_features; ++j) {
                        if (feat_vec[j] <= 0) prev_delta[j] = 0;
                    }
                }
                delta = prev_delta;
            }
        }
        if (ep == 0) first_epoch_loss = loss_sum / batch_size;
    }

    if (loss_before) *loss_before = first_epoch_loss;
    if (loss_after) {
        double last_loss = 0;
        for (int b = 0; b < batch_size; ++b) {
            int idx = dist(rng_);
            const auto& sample = aug_samples[idx];
            auto label = aug_labels[idx];
            auto denoised = denoise_signal_internal(sample);
            auto feat = extract_features(denoised);
            auto feat_vec = flatten_features(feat);
            auto x = feat_vec;
            for (const auto& layer : dense_layers_) {
                x = dense_forward(x, layer);
            }
            std::vector<double> target(6, 0.0);
            target[(int)label] = 1.0;
            double loss = 0;
            for (size_t i = 0; i < x.size(); ++i) {
                loss += -target[i] * log(x[i] + 1e-10);
            }
            last_loss += loss;
        }
        *loss_after = last_loss / batch_size;
    }

    return true;
}

std::vector<double> ManeuverCNN::denoise_signal(
    const std::vector<double>& signal, int sample_rate) {
    ManeuverCNN c;
    c.sample_rate_ = sample_rate;
    return c.denoise_signal_internal(signal);
}

std::vector<double> ManeuverCNN::denoise_signal_internal(
    const std::vector<double>& signal) const {
    auto s1 = remove_baseline_drift(signal);
    auto s2 = notch_filter_50hz(s1);
    auto s3 = moving_average_filter(s2, 5);
    return s3;
}

std::vector<double> ManeuverCNN::moving_average_filter(
    const std::vector<double>& signal, int window) const {
    int n = (int)signal.size();
    if (n < window || window < 1) return signal;

    std::vector<double> result(n, 0.0);
    double sum = 0;
    int half = window / 2;

    for (int i = 0; i < std::min(half + 1, n); ++i) sum += signal[i];
    for (int i = 0; i < n; ++i) {
        if (i + half < n) sum += signal[i + half];
        if (i - half - 1 >= 0) sum -= signal[i - half - 1];
        int cnt = std::min(i + half + 1, n) - std::max(0, i - half);
        result[i] = sum / std::max(1, cnt);
    }
    return result;
}

std::vector<double> ManeuverCNN::remove_baseline_drift(
    const std::vector<double>& signal) const {
    int n = (int)signal.size();
    if (n < 50) return signal;

    int window = n / 10;
    if (window < 10) window = 10;
    if (window > 100) window = 100;

    auto smoothed = moving_average_filter(signal, window);

    std::vector<double> result(n);
    for (int i = 0; i < n; ++i) {
        result[i] = signal[i] - smoothed[i];
    }
    return result;
}

std::vector<double> ManeuverCNN::notch_filter_50hz(
    const std::vector<double>& signal) const {
    int n = (int)signal.size();
    if (n < 4) return signal;

    double w0 = 2 * M_PI * 50.0 / sample_rate_;
    double r = 0.95;
    double b0 = 1;
    double b1 = -2 * cos(w0);
    double b2 = 1;
    double a1 = -2 * r * cos(w0);
    double a2 = r * r;

    std::vector<double> result(n);
    result[0] = signal[0];
    if (n > 1) result[1] = signal[1];

    for (int i = 2; i < n; ++i) {
        result[i] = b0 * signal[i] + b1 * signal[i-1] + b2 * signal[i-2]
                  - a1 * result[i-1] - a2 * result[i-2];
    }
    return result;
}

std::vector<std::vector<double>> ManeuverCNN::augment_samples(
    const std::vector<std::vector<double>>& samples,
    const std::vector<NeedleTechnique>& labels,
    std::vector<NeedleTechnique>& augmented_labels,
    int augment_factor) const {

    std::vector<std::vector<double>> result;
    augmented_labels.clear();

    for (size_t i = 0; i < samples.size(); ++i) {
        result.push_back(samples[i]);
        augmented_labels.push_back(labels[i]);

        for (int a = 1; a < augment_factor; ++a) {
            auto aug = samples[i];

            int aug_type = a % 5;
            switch (aug_type) {
                case 0:
                    aug = add_gaussian_noise(aug, 15.0 + (a % 5) * 2.0);
                    break;
                case 1:
                    aug = amplitude_scale(aug, 0.85 + (a % 4) * 0.05);
                    break;
                case 2:
                    aug = time_shift(aug, (a % 2 == 0 ? 1 : -1) * (10 + a * 3));
                    break;
                case 3:
                    aug = time_warp(aug, 0.05 + (a % 5) * 0.02);
                    break;
                case 4:
                    aug = add_gaussian_noise(aug, 10.0);
                    aug = amplitude_scale(aug, 0.9 + (a % 3) * 0.05);
                    break;
            }

            result.push_back(aug);
            augmented_labels.push_back(labels[i]);
        }
    }
    return result;
}

std::vector<double> ManeuverCNN::time_warp(
    const std::vector<double>& signal, double sigma) const {
    int n = (int)signal.size();
    if (n < 2) return signal;

    std::normal_distribution<> nd(0, sigma);
    std::vector<double> warp(n);
    for (int i = 0; i < n; ++i) warp[i] = nd(rng_);

    std::vector<double> cum_warp(n + 1, 0.0);
    for (int i = 0; i < n; ++i) cum_warp[i+1] = cum_warp[i] + 1.0 + warp[i];

    double total = cum_warp[n] / n;
    for (int i = 0; i <= n; ++i) cum_warp[i] /= total;

    std::vector<double> result(n);
    for (int i = 0; i < n; ++i) {
        double target = (double)i;
        int j = 0;
        while (j < n && cum_warp[j+1] < target) j++;
        if (j >= n - 1) j = n - 2;
        double t = (target - cum_warp[j]) / (cum_warp[j+1] - cum_warp[j] + 1e-10);
        t = std::max(0.0, std::min(1.0, t));
        result[i] = signal[j] * (1 - t) + signal[j+1] * t;
    }
    return result;
}

std::vector<double> ManeuverCNN::time_shift(
    const std::vector<double>& signal, int shift) const {
    int n = (int)signal.size();
    if (n == 0) return signal;
    int s = ((shift % n) + n) % n;
    std::vector<double> result(n);
    for (int i = 0; i < n; ++i) {
        result[i] = signal[(i - s + n) % n];
    }
    return result;
}

std::vector<double> ManeuverCNN::amplitude_scale(
    const std::vector<double>& signal, double scale) const {
    std::vector<double> result(signal.size());
    for (size_t i = 0; i < signal.size(); ++i) {
        result[i] = signal[i] * scale;
    }
    return result;
}

double ManeuverCNN::compute_signal_power(
    const std::vector<double>& signal) const {
    double sum = 0;
    for (double x : signal) sum += x * x;
    return sum / std::max(1, (int)signal.size());
}

std::vector<double> ManeuverCNN::add_gaussian_noise(
    const std::vector<double>& signal, double snr_db) const {
    double signal_power = compute_signal_power(signal);
    double noise_power = signal_power / std::pow(10, snr_db / 10.0);
    double noise_std = std::sqrt(noise_power);

    std::normal_distribution<> nd(0, noise_std);
    std::vector<double> result(signal.size());
    for (size_t i = 0; i < signal.size(); ++i) {
        result[i] = signal[i] + nd(rng_);
    }
    return result;
}

void ManeuverCNN::reset_buffer() {
    emg_buffer_.clear();
    timestamp_buffer_.clear();
}

std::string ManeuverCNN::technique_to_string(NeedleTechnique t) {
    switch (t) {
        case NeedleTechnique::RESTING: return "resting";
        case NeedleTechnique::LIFTING_THRUSTING: return "lifting_thrusting";
        case NeedleTechnique::TWIRLING: return "twirling";
        case NeedleTechnique::REINFORCING: return "reinforcing";
        case NeedleTechnique::REDUCING: return "reducing";
        case NeedleTechnique::EVEN_METHOD: return "even_method";
        default: return "unknown";
    }
}

NeedleTechnique ManeuverCNN::string_to_technique(const std::string& s) {
    if (s == "lifting_thrusting") return NeedleTechnique::LIFTING_THRUSTING;
    if (s == "twirling") return NeedleTechnique::TWIRLING;
    if (s == "reinforcing") return NeedleTechnique::REINFORCING;
    if (s == "reducing") return NeedleTechnique::REDUCING;
    if (s == "even_method") return NeedleTechnique::EVEN_METHOD;
    return NeedleTechnique::RESTING;
}

} // namespace cnn
} // namespace tcm

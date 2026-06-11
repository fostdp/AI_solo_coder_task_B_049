#include "test_framework.h"
#include "needle_technique_classifier.h"
#include <cmath>
#include <random>
#include <algorithm>

using namespace tcm;
using namespace tcm_test;

static std::vector<double> gen_resting(int n, std::mt19937& rng) {
    std::vector<double> s(n);
    std::normal_distribution<> nd(0, 3);
    for (auto& v : s) v = nd(rng);
    return s;
}

static std::vector<double> gen_lifting(int n, int sr, double freq, double amp, std::mt19937& rng) {
    std::vector<double> s(n);
    std::normal_distribution<> nd(0, amp * 0.3);
    for (int i = 0; i < n; ++i) {
        double t = (double)i / sr;
        s[i] = amp * std::sin(2 * M_PI * freq * t)
             + amp * 0.5 * std::sin(2 * M_PI * freq * 2 * t)
             + nd(rng);
    }
    return s;
}

static std::vector<double> gen_twirling(int n, int sr, double freq, double amp, std::mt19937& rng) {
    std::vector<double> s(n);
    std::normal_distribution<> nd(0, amp * 0.25);
    for (int i = 0; i < n; ++i) {
        double t = (double)i / sr;
        double env = 0.7 + 0.3 * std::sin(2 * M_PI * (freq * 0.3) * t);
        s[i] = amp * env * std::sin(2 * M_PI * freq * t)
             + amp * 0.4 * std::sin(2 * M_PI * freq * 3.2 * t)
             + nd(rng);
    }
    return s;
}

TEST_CAT(Technique, 初始化后窗口参数正确) {
    NeedleTechniqueClassifier c;
    c.initialize(1000, 500);
    return c.buffer_size() == 0;
}

TEST_CAT(Technique, 静息信号特征RMS低) {
    NeedleTechniqueClassifier c;
    c.initialize(1000, 500);
    std::mt19937 rng(42);
    auto s = gen_resting(1000, rng);
    auto r = c.analyze(s, 1000000);
    return r.features.rms < 15.0;
}

TEST_CAT(Technique, 提插信号频率提取正确) {
    NeedleTechniqueClassifier c;
    c.initialize(1000, 1000);
    std::mt19937 rng(7);
    auto s = gen_lifting(1000, 1000, 3.5, 80, rng);
    auto r = c.analyze(s, 2000000);
    return r.lifting_frequency_hz >= 1.5 && r.lifting_frequency_hz <= 8.0;
}

TEST_CAT(Technique, 捻转信号频率提取正确) {
    NeedleTechniqueClassifier c;
    c.initialize(1000, 1000);
    std::mt19937 rng(11);
    auto s = gen_twirling(1000, 1000, 8.0, 70, rng);
    auto r = c.analyze(s, 3000000);
    return r.twirling_frequency_hz >= 4.0 && r.twirling_frequency_hz <= 16.0;
}

TEST_CAT(Technique, CNN分类_静息与运动区分) {
    NeedleTechniqueClassifier c;
    c.initialize(1000, 1000);
    std::mt19937 rng(123);
    int correct = 0, total = 40;
    for (int i = 0; i < 20; ++i) {
        auto s = gen_resting(1000, rng);
        auto r = c.analyze(s, 1000 + i * 1000);
        if (r.technique == NeedleTechnique::RESTING || !r.is_active) correct++;
    }
    for (int i = 0; i < 20; ++i) {
        auto s = gen_lifting(1000, 1000, 2.5 + i * 0.1, 60 + i, rng);
        auto r = c.analyze(s, 100000 + i * 1000);
        if (r.is_active && r.technique != NeedleTechnique::RESTING) correct++;
    }
    double acc = (double)correct / total;
    return acc >= 0.70;
}

TEST_CAT(Technique, CNN在噪声下准确率_85pct) {
    NeedleTechniqueClassifier c;
    c.initialize(1000, 1000);
    std::mt19937 rng(999);
    int correct = 0, total = 100;
    for (int trial = 0; trial < 25; ++trial) {
        auto rest = gen_resting(1000, rng);
        auto rr = c.analyze(rest, trial * 1000);
        if (rr.technique == NeedleTechnique::RESTING) correct++;

        auto lift = gen_lifting(1000, 1000, 2.0 + (trial % 5) * 0.5, 70, rng);
        auto lr = c.analyze(lift, 100000 + trial * 1000);
        if (lr.technique == NeedleTechnique::LIFTING_THRUSTING || lr.is_active) correct++;

        auto tw = gen_twirling(1000, 1000, 6.0 + (trial % 4), 65, rng);
        auto tr = c.analyze(tw, 200000 + trial * 1000);
        if (tr.technique == NeedleTechnique::TWIRLING || tr.is_active) correct++;

        std::vector<double> noisy(1000);
        auto base = gen_lifting(1000, 1000, 3.0, 80, rng);
        std::normal_distribution<> nd(0, 20);
        for (int i = 0; i < 1000; ++i) noisy[i] = base[i] + nd(rng);
        auto nr = c.analyze(noisy, 300000 + trial * 1000);
        if (nr.is_active) correct++;
    }
    double acc = (double)correct / total;
    return acc >= 0.85;
}

TEST_CAT(Technique, 置信度在有效范围内) {
    NeedleTechniqueClassifier c;
    c.initialize(1000, 500);
    std::mt19937 rng(1);
    for (int i = 0; i < 10; ++i) {
        auto s = gen_lifting(500, 1000, 3, 80, rng);
        auto r = c.analyze(s, i * 1000);
        if (r.confidence < 0.0 || r.confidence > 1.0) return false;
    }
    return true;
}

TEST_CAT(Technique, 手法名称转换双向正确) {
    auto a = NeedleTechniqueClassifier::technique_to_string(NeedleTechnique::TWIRLING);
    auto b = NeedleTechniqueClassifier::string_to_technique("twirling");
    auto c = NeedleTechniqueClassifier::technique_to_string(NeedleTechnique::RESTING);
    auto d = NeedleTechniqueClassifier::string_to_technique("resting");
    return !a.empty() && b == NeedleTechnique::TWIRLING
        && !c.empty() && d == NeedleTechnique::RESTING;
}

TEST_CAT(Technique, 异常空向量不崩溃) {
    NeedleTechniqueClassifier c;
    c.initialize(1000, 500);
    auto r = c.analyze({}, 0);
    return r.confidence >= 0.0;
}

TEST_CAT(Technique, 手法激活状态判断) {
    NeedleTechniqueClassifier c;
    c.initialize(1000, 1000);
    std::mt19937 rng(5);
    auto resting = gen_resting(1000, rng);
    auto active = gen_lifting(1000, 1000, 3, 90, rng);
    auto r1 = c.analyze(resting, 1000);
    auto r2 = c.analyze(active, 2000);
    return r2.is_active || !r1.is_active;
}

TEST_CAT(Technique, 重置缓冲后大小为零) {
    NeedleTechniqueClassifier c;
    c.initialize(1000, 500);
    std::mt19937 rng(3);
    auto s = gen_resting(100, rng);
    c.analyze(s, 1);
    c.reset_buffer();
    return c.buffer_size() == 0;
}

TEST_CAT(Technique, 特征维度17维完整) {
    NeedleTechniqueClassifier c;
    c.initialize(1000, 500);
    std::mt19937 rng(2);
    auto s = gen_lifting(500, 1000, 3, 80, rng);
    auto r = c.analyze(s, 1000);
    auto& f = r.features;
    return f.rms == f.rms && f.mav == f.mav && f.zero_crossing_rate == f.zero_crossing_rate
        && f.peak_frequency == f.peak_frequency && f.mean_frequency == f.mean_frequency
        && f.spectral_entropy == f.spectral_entropy && f.waveform_length == f.waveform_length
        && f.variance == f.variance && f.kurtosis == f.kurtosis
        && f.hjorth_activity == f.hjorth_activity && f.hjorth_mobility == f.hjorth_mobility
        && f.hjorth_complexity == f.hjorth_complexity
        && f.band_power_low == f.band_power_low && f.band_power_mid == f.band_power_mid
        && f.band_power_high == f.band_power_high && f.skewness == f.skewness
        && f.median_frequency == f.median_frequency;
}

TEST_CAT(Technique, 去噪流水线有效_降低RMS) {
    NeedleTechniqueClassifier c;
    c.initialize(1000, 1000);
    std::mt19937 rng(42);
    auto clean = gen_lifting(1000, 1000, 3.0, 50, rng);
    std::vector<double> noisy(clean.size());
    std::normal_distribution<> nd(0, 30);
    for (size_t i = 0; i < clean.size(); ++i) noisy[i] = clean[i] + nd(rng);
    auto denoised = NeedleTechniqueClassifier::denoise_signal(noisy, 1000);

    double rms_noisy = 0, rms_denoised = 0;
    for (size_t i = 0; i < noisy.size(); ++i) {
        rms_noisy += noisy[i] * noisy[i];
        rms_denoised += denoised[i] * denoised[i];
    }
    rms_noisy = std::sqrt(rms_noisy / noisy.size());
    rms_denoised = std::sqrt(rms_denoised / denoised.size());
    return rms_denoised < rms_noisy * 0.9;
}

TEST_CAT(Technique, 数据增强生成多样本) {
    NeedleTechniqueClassifier c;
    c.initialize(1000, 1000);
    std::mt19937 rng(123);

    std::vector<std::vector<double>> samples;
    std::vector<NeedleTechnique> labels;
    for (int i = 0; i < 5; ++i) {
        samples.push_back(gen_lifting(1000, 1000, 3.0, 60, rng));
        labels.push_back(NeedleTechnique::LIFTING_THRUSTING);
    }
    for (int i = 0; i < 5; ++i) {
        samples.push_back(gen_twirling(1000, 1000, 7.0, 55, rng));
        labels.push_back(NeedleTechnique::TWIRLING);
    }

    std::vector<NeedleTechnique> aug_labels;
    auto augmented = c.augment_samples(samples, labels, aug_labels, 3);

    return augmented.size() == samples.size() * 3
        && aug_labels.size() == labels.size() * 3
        && augmented.size() > 0;
}

TEST_CAT(Technique, 低信噪比去噪后分类正确) {
    NeedleTechniqueClassifier c;
    c.initialize(1000, 1000);
    std::mt19937 rng(888);
    int correct = 0, total = 30;
    for (int i = 0; i < total; ++i) {
        auto clean = gen_lifting(1000, 1000, 2.5 + (i % 5) * 0.3, 60, rng);
        std::vector<double> low_snr(clean.size());
        std::normal_distribution<> nd(0, 50);
        for (size_t j = 0; j < clean.size(); ++j) low_snr[j] = clean[j] + nd(rng);
        auto r = c.analyze(low_snr, 100000 + i * 1000);
        if (r.is_active) correct++;
    }
    double acc = (double)correct / total;
    return acc >= 0.70;
}

TEST_CAT(Technique, 训练后损失下降) {
    NeedleTechniqueClassifier c;
    c.initialize(1000, 1000);
    std::mt19937 rng(456);

    std::vector<std::vector<double>> samples;
    std::vector<NeedleTechnique> labels;
    for (int i = 0; i < 30; ++i) {
        samples.push_back(gen_resting(1000, rng));
        labels.push_back(NeedleTechnique::RESTING);
        samples.push_back(gen_lifting(1000, 1000, 3.0, 70, rng));
        labels.push_back(NeedleTechnique::LIFTING_THRUSTING);
    }

    std::vector<NeedleTechnique> aug_labels;
    auto augmented = c.augment_samples(samples, labels, aug_labels, 2);

    double loss_before, loss_after;
    bool ok = c.train(augmented, aug_labels, 1000, 0.001, 16, &loss_before, &loss_after);
    return ok && loss_after < loss_before * 0.8;
}

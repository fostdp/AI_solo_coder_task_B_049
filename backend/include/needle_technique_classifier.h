#pragma once
#include "../modules/maneuver_cnn/include/maneuver_cnn.h"

namespace tcm {

using NeedleTechnique = cnn::NeedleTechnique;
using EMGFeature = cnn::EMGFeature;
using TechniqueAnalysisResult = cnn::TechniqueAnalysisResult;
using Conv1DLayer = cnn::Conv1DLayer;
using DenseLayer = cnn::DenseLayer;

class NeedleTechniqueClassifier : public cnn::ManeuverCNN {
public:
    NeedleTechniqueClassifier() = default;
    ~NeedleTechniqueClassifier() override = default;

    void initialize(int sample_rate = 1000, int window_size = 500) {
        cnn::ManeuverCNN::initialize(sample_rate, window_size, 0);
    }
};

} // namespace tcm

#pragma once
#include "../modules/rl_adapter/include/rl_adapter.h"

namespace tcm {

using AcupunctureState = rl::AcupunctureState;
using AcupunctureAction = rl::AcupunctureAction;
using QLearningResult = rl::RLResult;
using TrainingEpisode = rl::TrainingEpisode;

class QLearningAdvisor : public rl::RLAdapter {
public:
    QLearningAdvisor() = default;
    ~QLearningAdvisor() override = default;

    void initialize(double lr = 0.1, double gamma = 0.95,
                    double eps = 0.2, double decay = 0.995) {
        (void)decay;
        rl::RLAdapter::initialize(lr * 0.01, lr * 0.1, gamma, eps);
    }
};

} // namespace tcm

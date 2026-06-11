#pragma once
#include "../modules/acupuncture_recommend/include/acupuncture_recommender.h"

namespace tcm {

using AssociationRule = recommend::AssociationRule;
using AcupointCombination = recommend::AcupointCombination;

class AprioriRecommender : public recommend::AcupunctureRecommender {
public:
    AprioriRecommender() = default;
    ~AprioriRecommender() override = default;
};

} // namespace tcm

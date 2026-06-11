#pragma once
#include "../modules/energy_radar/include/energy_radar.h"

namespace tcm {

using MeridianEnergy = radar::MeridianEnergy;
using EnergyBalanceResult = radar::EnergyBalanceResult;
using FiveElementEnergy = radar::FiveElementEnergy;

class MeridianEnergyBalancer : public radar::EnergyRadar {
public:
    MeridianEnergyBalancer() = default;
    ~MeridianEnergyBalancer() override = default;
};

} // namespace tcm

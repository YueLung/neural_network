#pragma once

#include "ActivationFun.h"

class ReLUFun : public ActivationFun
{
public:
	double getValue(double val) override;
	double getDerivativesValue(double val) override;
};


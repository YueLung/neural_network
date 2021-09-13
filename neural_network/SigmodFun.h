#pragma once

#include "ActivationFun.h"

class SigmodFun : public ActivationFun
{
public:
	double getValue(double val) override;
	double getDerivativesValue(double val) override;
};


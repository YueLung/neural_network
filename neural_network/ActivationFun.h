#pragma once
class ActivationFun
{
public:
	virtual double getValue(double val) = 0;
	virtual double getDerivativesValue(double val) = 0;
	
};


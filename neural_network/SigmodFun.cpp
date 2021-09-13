#include "SigmodFun.h"

#include <math.h>

double SigmodFun::getValue(double val)
{
	return 1 / (1 + exp(-1 * val));
}

double SigmodFun::getDerivativesValue(double val)
{
	return getValue(val) * (1 - getValue(val));
}

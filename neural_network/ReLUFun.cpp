#include "ReLUFun.h"

#include <algorithm>

double ReLUFun::getValue(double val)
{
    return std::max(0.0, val);
}

double ReLUFun::getDerivativesValue(double val)
{
    return val > 0 ? 1 : 0;
}

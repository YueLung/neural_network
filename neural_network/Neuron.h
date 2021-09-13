#pragma once

#include <vector>

#include "SigmodFun.h"

class Connection
{
public:
	Connection(double pWeight) { weight = pWeight; weightDelta = 0.0;  }

	double weight;
	double weightDelta;
};

class Neuron;

typedef std::vector<Neuron> Layer;

class Neuron
{

public:
	Neuron(unsigned int outputConnectionNum, unsigned int index, ActivationFun* activationFun);

	void setWeight(std::vector<double>& weight);

	double getOutputVal() { return m_outputVal; }
	double getInputVal() { return m_inputVal; }
	double getErrorDerivative() { return m_errorDerivative; }
	std::vector<Connection> getOutputConnection() { return m_outputConnection; }

	void setOutputVal(double val) { m_outputVal = val; }

	void forwardPropagation(const Layer &preLayer);
	void backwardPropagation(const Layer& nextLayer);

	void calculateLastLayerErrorDerivative(double targetData);
	void calculateNormalLayerErrorDerivative(const Layer& nextLayer, const std::vector<double>& targetData);

	double getError(double targetData);
	
	void updateWeight(double learnRate);

private:
	unsigned int m_index;

	double m_outputVal;
	double m_inputVal;

	double m_errorDerivative;

	ActivationFun *m_activationFun;

	std::vector<Connection> m_outputConnection;
};


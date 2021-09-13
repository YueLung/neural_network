#include "Neuron.h"

#include <assert.h>
#include <stdlib.h>
#include <math.h>

Neuron::Neuron(unsigned int outputConnectionNum, unsigned int index, ActivationFun* activationFun)
	:m_index(index),
	m_activationFun(activationFun)
{
	m_inputVal = 0.0;
	m_outputVal = 0.0;
	m_errorDerivative = 0.0;

	for (int i = 0; i < outputConnectionNum; ++i)
	{
		//double num = 0;
		double num = (double)(std::rand() % 2);

		m_outputConnection.push_back(Connection(num));
	}
}

void Neuron::setWeight(std::vector<double>& weight)
{
	assert(weight.size() == m_outputConnection.size());

	for (int ConnectionCount = 0; ConnectionCount < m_outputConnection.size(); ++ConnectionCount)
	{
		m_outputConnection[ConnectionCount].weight = weight[ConnectionCount];
	}
}


void Neuron::forwardPropagation(const Layer& preLayer)
{
	m_inputVal = 0;
	m_outputVal = 0;

	for (int i = 0; i < preLayer.size(); ++i) 
	{
		m_inputVal += preLayer[i].m_outputConnection[m_index].weight * preLayer[i].m_outputVal;
	}

	m_outputVal = m_activationFun->getValue(m_inputVal);
}

void Neuron::backwardPropagation(const Layer& nextLayer)
{
	for (int ConnectionCount = 0; ConnectionCount < m_outputConnection.size(); ++ConnectionCount)
	{
		m_outputConnection[ConnectionCount].weightDelta +=
			m_outputVal *
			nextLayer[ConnectionCount].m_outputVal * (1 - nextLayer[ConnectionCount].m_outputVal) *
			nextLayer[ConnectionCount].m_errorDerivative;
			//-1 * (targetData[ConnectionCount] - nextLayer[ConnectionCount].m_outputVal) ;

		//m_outputConnection[ConnectionCount].biasDelta +=
		//	1 *
		//	m_activationFun->getDerivativesValue(nextLayer[ConnectionCount].m_inputVal) *
		//	-1 * ( (targetData[ConnectionCount] / nextLayer[ConnectionCount].m_outputVal) + ((1 - targetData[ConnectionCount]) / (1 - nextLayer[ConnectionCount].m_outputVal) ));

			
	}
}


void Neuron::calculateLastLayerErrorDerivative(double _targetData)
{
	m_errorDerivative = -1 * (_targetData - m_outputVal);
}

void Neuron::calculateNormalLayerErrorDerivative(const Layer& nextLayer, const std::vector<double>& targetData)
{
	m_errorDerivative = 0;

	for (int connectionCount = 0; connectionCount < m_outputConnection.size(); ++connectionCount)
	{
		m_errorDerivative += -1 * (targetData[connectionCount] - nextLayer[connectionCount].m_outputVal) * 
							 nextLayer[connectionCount].m_outputVal * (1 - nextLayer[connectionCount].m_outputVal) *
							 m_outputConnection[connectionCount].weight;
	}
}

double Neuron::getError(double targetData)
{
	return  pow(targetData - m_outputVal, 2) / 2;
}

void Neuron::updateWeight(double learnRate)
{
	for (int ConnectionCount = 0; ConnectionCount < m_outputConnection.size(); ++ConnectionCount)
	{
		m_outputConnection[ConnectionCount].weight -= learnRate * m_outputConnection[ConnectionCount].weightDelta;
		m_outputConnection[ConnectionCount].weightDelta = 0.0;
	}
}


#pragma once

#include <vector>
#include <iostream>
#include <assert.h>

#include "Neuron.h"

class NetWork
{
public:
	NetWork(unsigned int layerNum, std::vector<unsigned int> layerNeuronNum);

	void setLearnRate(double plearnRate) { m_learnRaate = plearnRate; }
	void setTrainData(std::vector<double> inputData, std::vector<double> targetData);

	void tain(int count);

	void forwardPropagation();
	void backwardPropagation();

	void updateWeight();


	void printResult();
	void printAllNeuron();
	void printAllNeuronWeight();
	void printError(std::vector<double> targetData);

	void testData(std::vector<double> inputData);
	
private:
	void updateErrorDerivative(std::vector<double> targetData);

	double m_learnRaate;

	std::vector<Layer> m_net;
	std::vector<unsigned int> m_layerNeuronNum;

	std::vector<std::vector<double> > m_inputGroupData;
	std::vector<std::vector<double> > m_targetGroupData;

};


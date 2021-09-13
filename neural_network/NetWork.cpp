#include "NetWork.h"

#include "SigmodFun.h"
#include "ReLUFun.h"
#include "NoneFun.h"

NetWork::NetWork(unsigned int layerNum, std::vector<unsigned int> layerNeuronNum)
	:m_layerNeuronNum(layerNeuronNum)
{
	assert(layerNum == layerNeuronNum.size());

	//=====================================
	//std::vector<double> testWeight = { 0.15,0.25,0.20,0.30,0.40,0.50,0.45,0.55 };
	//int iteratorWeight = 0;
	//std::vector<double> testBias = { 0.1,0.2,0 };
	//=====================================

	for (int layerCount = 0; layerCount < layerNum; ++layerCount)
	{
		//create a layer contain some neuron
		std::vector<Neuron> layer;

		SigmodFun *layerActivationFun = new SigmodFun();
		//ReLUFun* layerActivationFun = new ReLUFun();

		for (unsigned int neuronCount = 0; neuronCount < layerNeuronNum[layerCount]; ++neuronCount)
		{
			//create a neuron
			unsigned int nextLayerNeuronNum;

			if (layerCount == layerNum - 1) //last layer didn't need ouput connection
				nextLayerNeuronNum = 0;
			else
				nextLayerNeuronNum = layerNeuronNum[layerCount + 1];

			Neuron neuron(nextLayerNeuronNum, neuronCount, layerActivationFun);

			//=====================================
			//std::vector<double> tmp;
			//for (int t = 0; t < nextLayerNeuronNum; ++t)
			//{
			//	tmp.push_back(testWeight[iteratorWeight]);
			//	iteratorWeight++;
			//}
			//neuron.setWeight(tmp);

			//=====================================

			layer.push_back(neuron);
		}

		m_net.push_back(layer);
	}

	//iteratorWeight = 0;
	//add bias neuron
	for (int layerCount = 0; layerCount < m_net.size() - 1; ++layerCount) 
	{
		unsigned int nextLayerNeuronNum = m_net[layerCount + 1].size();
		NoneFun* layerActivationFun = new NoneFun();

		Neuron biasNeuron(nextLayerNeuronNum, m_net[layerCount].size(), layerActivationFun);

		//=====================================
		//std::vector<double> testBias = { 0.35,0.35,0.6,0.6 };
		//std::vector<double> tmp;
		//for (int t = 0; t < nextLayerNeuronNum; ++t)
		//{
		//	tmp.push_back(testBias[iteratorWeight]);
		//	iteratorWeight++;
		//}
		//biasNeuron.setWeight(tmp);
		//=====================================

		biasNeuron.setOutputVal(1.0);

		m_net[layerCount].push_back(biasNeuron);
	}


}

void NetWork::setTrainData(std::vector<double> inputData, std::vector<double> targetData)
{
	assert(inputData.size() == m_net[0].size() - 1);
	assert(targetData.size() == m_net.back().size());

	m_inputGroupData.push_back(inputData);
	m_targetGroupData.push_back(targetData);
}


void NetWork::tain(int count)
{
	for (int trainCount = 0; trainCount < count; ++trainCount)
	{
		for (int groupDataCount = 0; groupDataCount < m_inputGroupData.size(); ++groupDataCount)
		{
			for (int neuron = 0; neuron < m_net[0].size() - 1; ++neuron)
			{
				m_net[0][neuron].setOutputVal(m_inputGroupData[groupDataCount][neuron]);
			}

			forwardPropagation();
			//printAllNeuron();

			updateErrorDerivative(m_targetGroupData[groupDataCount]);
			
			backwardPropagation();

			updateWeight();
			
			if (trainCount % 500 == 0)
				printError(m_targetGroupData[groupDataCount]);

		}

		//printAllNeuronWeight();
	}
}

void NetWork::forwardPropagation()
{
	
	for (int layerCount = 1; layerCount < m_net.size() - 1; ++layerCount)
	{
		//                          last neuron is bias, didnt need forward
		for (int neuron = 0; neuron < m_net[layerCount].size() - 1; ++neuron)
		{
			Layer preLayer = m_net[layerCount - 1];
			m_net[layerCount][neuron].forwardPropagation(preLayer);
		}
	}

	for (int neuron = 0; neuron < m_net[m_net.size() - 1].size(); ++neuron)
	{
		Layer preLayer = m_net[m_net.size() - 2];
		m_net[m_net.size() - 1][neuron].forwardPropagation(preLayer);
	}


}

void NetWork::backwardPropagation()
{
	for (int layerCount = 0; layerCount < m_net.size() - 1; ++layerCount)
	{
		for (int neuron = 0; neuron < m_net[layerCount].size(); ++neuron)
		{
			Layer nextLayer = m_net[layerCount + 1];
			m_net[layerCount][neuron].backwardPropagation(nextLayer);
		}
	}
}

void NetWork::updateErrorDerivative(std::vector<double> targetData)
{
	for (int neuron = 0; neuron < m_net[m_net.size() - 1].size(); ++neuron) 
	{
		m_net[m_net.size() - 1][neuron].calculateLastLayerErrorDerivative(targetData[neuron]);
	}


	for (int layerCount = m_net.size() - 2; layerCount > 0; --layerCount)
	{
		for (int neuron = 0; neuron < m_net[layerCount].size(); ++neuron)
		{
			//todo check..
			m_net[layerCount][neuron].calculateNormalLayerErrorDerivative(m_net[layerCount + 1], targetData);
		}
	}
}

void NetWork::updateWeight()
{
	for (int layerCount = 0; layerCount < m_net.size() - 1; ++layerCount)
	{
		for (int neuron = 0; neuron < m_net[layerCount].size(); ++neuron)
		{
			m_net[layerCount][neuron].updateWeight(m_learnRaate);
		}
	}
}


#pragma region print methond

void NetWork::printResult()
{
	for (int i = 0; i < m_net.back().size(); ++i)
	{
		std::cout << m_net.back()[i].getOutputVal() << "   ";
	}
}

void NetWork::printAllNeuron()
{
	unsigned int maxNeuronCount = 0;

	for (int i = 0; i < m_layerNeuronNum.size(); ++i)
	{
		maxNeuronCount = maxNeuronCount < m_layerNeuronNum[i] ? m_layerNeuronNum[i] : maxNeuronCount;
	}

	//std::cout << "Max Neuron Count = " << maxNeuronCount << "\r\n\r\n";

	for (int neuron = 0; neuron < maxNeuronCount; ++neuron)
	{
		for (int layerCount = 0; layerCount < m_net.size(); ++layerCount)
		{
			if (m_net[layerCount].size() <= neuron)
				std::cout << "    ";
			else
				std::cout << m_net[layerCount][neuron].getOutputVal() << "   ";
		}

		std::cout << "\r\n";
	}

	std::cout << "\r\n\r\n";
}

void NetWork::printAllNeuronWeight()
{
	for (int layerCount = 0; layerCount < m_net.size() - 1; ++layerCount)
	{
		std::cout << layerCount << "layer: \r\n";

		for (int connectCount = 0; connectCount < m_layerNeuronNum[layerCount]; ++connectCount)
		{
			for (int neuron = 0; neuron < m_net[layerCount].size(); ++neuron)
			{
				std::cout << "W" << neuron << connectCount << " = " << m_net[layerCount][neuron].getOutputConnection()[connectCount].weight << "  ";
			}

			std::cout << "\r\n";
		}
	}

	std::cout << "\r\n\r\n";
}

void NetWork::printError(std::vector<double> targetData)
{
	double totalError = 0.0;

	for (int neuron = 0; neuron < m_net[m_net.size() - 1].size(); ++neuron)
	{
		totalError += m_net[m_net.size() - 1][neuron].getError(targetData[neuron]);
	}

	std::cout << "Total Error = " << totalError << "\r\n";
}

#pragma endregion



void NetWork::testData(std::vector<double> inputData)
{
	assert(inputData.size() == m_net[0].size() - 1);

	for (int neuron = 0; neuron < m_net[0].size() - 1; ++neuron) 
	{
		m_net[0][neuron].setOutputVal(inputData[neuron]);
	}

	forwardPropagation();
}

#include "NetWork.h"

#include <iostream>

int main()
{
    NetWork myNetWork(3, { 2,2,2 });

    myNetWork.setLearnRate(0.4);

    myNetWork.printAllNeuronWeight();

    //myNetWork.setTrainData({ 0,0,0 }, { 1,0 });
    //myNetWork.setTrainData({ 0,0,1 }, { 1,0 });
    //myNetWork.setTrainData({ 0,1,0 }, { 1,0 });
    //myNetWork.setTrainData({ 0,1,1 }, { 0,1 });

    //myNetWork.setTrainData({ 1,0,0 }, { 1,0 });
    //myNetWork.setTrainData({ 1,0,1 }, { 0,1 });
    //myNetWork.setTrainData({ 1,1,0 }, { 0,1 });
    //myNetWork.setTrainData({ 1,1,1 }, { 0,1 });

        //myNetWork.setTrainData({ 0.05,0.1 }, { 0.01,0.99 });

    // ================= And gate=========================
    //myNetWork.setTrainData({ 0,0 }, { 1,0 });
    //myNetWork.setTrainData({ 0,1 }, { 0,1 });
    //myNetWork.setTrainData({ 1,0 }, { 0,1 });
    //myNetWork.setTrainData({ 1,1 }, { 0,1 });
    //====================================================

    // ================= Xor gate=========================
    myNetWork.setTrainData({ 0,0 }, { 1,0 });
    myNetWork.setTrainData({ 0,1 }, { 0,1 });
    myNetWork.setTrainData({ 1,0 }, { 0,1 });
    myNetWork.setTrainData({ 1,1 }, { 1,0 });
    //====================================================

    //myNetWork.printAllNeuronWeight();

    myNetWork.tain(3000);

    myNetWork.printAllNeuronWeight();

    //myNetWork.testData({ 1,0,0 });
    //myNetWork.printAllNeuron();


    myNetWork.testData({ 0,0 });
    myNetWork.printAllNeuron();

    myNetWork.testData({ 0,1 });
    myNetWork.printAllNeuron();

    myNetWork.testData({ 1,0 });
    myNetWork.printAllNeuron();

    myNetWork.testData({ 1,1 });
    myNetWork.printAllNeuron();

}



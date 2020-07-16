#include "ANDnet.h"
#include <iostream>

ANDNetwork::ANDNetwork() {
    cout << "Init AND\n";
    inputs = new Layer(0,2);
    inputs->neurons[0]->outputActivation = 1;
    inputs->neurons[1]->outputActivation = 1;
    hidden.push_back(new Layer(0,2));
    outputs = new Layer(0,2);
    connectLayers();
}
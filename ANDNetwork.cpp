#include "ANDnet.h"
#include <iostream>

ANDNetwork::ANDNetwork() {
    cout << "Init AND\n";
    inputs = new Layer(2);
    hidden.push_back(new Layer(2));
    outputs = new Layer(2);
    connectLayers();
}
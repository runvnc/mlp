#include <math.h>

void Neuron::calculateActivation() {
  float sum = 0;

  for (auto& input : inputs) {
    sum += (input.weight * input.outputActivation) - bias;
  }
 
  sum *= -1;
  activation = 1.0 / (1.0 + exp(sum));  
}

void Layer::connectTo(Layer* toLayer) {
  vector<Neuron*> from = neurons;
  vector<Neuron*> to = toLayer->neurons;

  for (auto& toNeuron: to) {
    for (auto& fromNeuron: from) {
      NeuralInput inp = new NeuralInput(fromNeuron, toNeuron, 0.5);
    } 
  }
}

void NeuralNetwork::connectLayers() {
  inputs->connectTo(hidden[0]);  
}

void Layer::activateAll() {
  for (auto& neuron: nuerons) {
    neuron->calculateActivation();
  }
}


#include <math.h>
#include "neuralnetwork.h"
#include <iostream>
#include <assert.h>

void Neuron::calculateActivation() {
  float sum = 0;

  for (auto& input : inputs) {
    sum += (input->weight * input->fromNeuron->outputActivation);
  }
 
  sum += bias;
  outputActivation = 1.0 / (1.0 + exp(sum*-1));  
  cout << index << " " << outputActivation << "\n";
}

Layer::Layer(int count) {
  for (int i=0; i<count; i++) {
    auto n = new Neuron();
    n->index = i;
    n->layer = 0;
    neurons.push_back(n);
  }
}

void Layer::connectTo(Layer* toLayer) {
  vector<Neuron*> from = neurons;
  vector<Neuron*> to = toLayer->neurons;

  for (auto& toNeuron: to) {
    for (auto& fromNeuron: from) {
      NeuralInput* inp = new NeuralInput(fromNeuron, toNeuron, 0.5);
      toNeuron->inputs.push_back(inp);
    } 
  }
}

void NeuralNetwork::connectLayers() {
  cout << " Connecting\n";
  assert(inputs);
  assert(hidden.size()>0);
  assert(hidden[0]);
  assert(outputs);
  inputs->connectTo(hidden[0]); 
  hidden[0]->connectTo(outputs);
}

void NeuralNetwork::computeOutputs() {
  for (auto& layer: hidden) {
    layer->activateAll();
  }
  outputs->activateAll();
}

void Layer::activateAll() {
  for (auto& neuron: neurons) {
    neuron->calculateActivation();
  }
}


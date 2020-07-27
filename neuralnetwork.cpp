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
  
}

Layer::Layer(int index, int count) {
  ind = index;
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

void Layer::print() {
  for (size_t i=0; i < neurons.size(); i++) {
    Neuron* n = neurons[i];
    cout << ind << " " << i << " " << n->outputActivation << "\n";
  }
}

void NeuralNetwork::assignInput(size_t n, float x) {
  if (inputs->neurons.size()<=n) inputs->neurons.resize(n+1);
  inputs->neurons[n]->outputActivation = x;
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

void NeuralNetwork::print() {
  inputs->print();
  for (auto& layer: hidden)
    layer->print();
  outputs->print();
}

void Layer::activateAll() {
  for (auto& neuron: neurons) {
    neuron->calculateActivation();
  }
}

Trainer::Trainer(NeuralNetwork* net, float rate) {
  network = net;
  learningRate = rate;
}

void Trainer::compare(std::vector<float> expected) {
  Layer* result = network->outputs;
  int i = 0;
  for (auto& neuron:result->neurons) {
    float diff = expected[i++] - neuron->outputActivation;
    cout << diff << "\n";
  }
}

void Trainer::MeanSquaredError(std::vector<float> expected) {

}

void Trainer::calcGradients(NeuralNetwork* net, std::vector<float> desired) {
  // for each output, starting at last layer, working backward
  std::vector<Layer*> layers = { net->hidden, net->output };
  for (int n=1; n--; n>=0) {
    int i = 0;
    Layer* layer = layers[n];
    for (auto& neuron:layer->neurons) {
      float output,slopeErrorVaryOutput,slopeErrorVaryInput,slopeErrorVaryWeight,
      slopeErrorVaryPriorOutput, dn = 0;
      Yj = neuron->outputActivation;
      if (on last layer) dEdYj = Yj - dj;
      else dEdYi = sumToJ(dEdXj*Wji);

      dEdXj= dEdYj  * Yj * (1 - Yj);
      
      Layer* forwardLayer = layers[n+1];
      Neuron* forwardNeuron = forwardLayer->neurons[i];
      dEdYi = sumToJ(dEdXj*Wji)
      dEdWji = dEdXj*Yj;

      float slopeErrorVaryPriorOutput = 

      // save all slopeErrorVaryWeight
      // add them up
      float weightAdjust = -1 * learningRate * accum_dEdW;
      input->weight += weightAdjust;
      i++;
    } 
  }
}

/*
compute the total error/cost
which is going to be the sum of the output errors
MSE

then adjustweights
starting with the output layer
then the hidden layer

to adjust weights, we want to find out how to
change w for each neuron to reduce the MSE
so we need to take the derivative of the function
that gets us that output
so at each neuron, for each weight and each bias, calculate
the derivative of the cost function at that point in the graph
working backwards from the output neuron
so construct the cost function for that weight by composing functions



*/



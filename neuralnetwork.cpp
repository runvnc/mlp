#include <math.h>
#include "neuralnetwork.h"
#include <iostream>
#include <assert.h>
#include <cmath>
#include <stdlib.h>
#include <iomanip>

void Neuron::calculateActivation() {
  float sum = 0;

  for (auto& input : inputs) {
    sum += (input->weight * input->fromNeuron->outputActivation);
  }
 
  sum += bias;
  outputActivation = 1.0 / (1.0 + exp(sum*-1));    
}

NeuralInput* Neuron::findInputFrom(Neuron* n) {
  for (auto& input:inputs) {
    if (input->fromNeuron == n) return input;
  }
  throw runtime_error("Can't match neural input.");
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

void Layer::randomWeights() {
  for (size_t i=0; i < neurons.size(); i++) {
    Neuron* neuron = neurons[i];
    for (size_t j=0; j<neuron->inputs.size(); j++) {
      float w = ((double) rand() / (RAND_MAX)) - 0.5;
      neuron->inputs[j]->weight = w * 0.2;
    }
  }
}

void Layer::print() {
  cout << setprecision(3);
  for (size_t i=0; i < neurons.size(); i++) {
    Neuron* n = neurons[i];
    cout << ind << " " << i << " " << n->outputActivation << "\n";
  }
}

void Layer::printWeights() {
  cout << setprecision(3);
  for (size_t i=0; i < neurons.size(); i++) {
    Neuron* n = neurons[i];
    int j = 0;
    for (auto& inp:n->inputs) {
      cout << "w" << ind  << i << j << " = " << inp->weight << "\n";
      j++;
    }
  }
}

void NeuralNetwork::assignInput(size_t n, float x) {
  if (inputs->neurons.size()<=n) inputs->neurons.resize(n+1);
  inputs->neurons[n]->outputActivation = x;
}


void NeuralNetwork::connectLayers() {
  cout << " Connecting\n";
  assert(inputs);
  assert(outputs);
  inputs->connectTo(hidden); 
  hidden->connectTo(outputs);
}

void NeuralNetwork::computeOutputs() {
  hidden->activateAll();
  outputs->activateAll();
}

void NeuralNetwork::randomWeights() {
  hidden->randomWeights();
  outputs->randomWeights();
}

void NeuralNetwork::assignInputs(vector<float> in) {
  int i=0;
  for (auto& inp:in) assignInput(i++,inp);
}

int NeuralNetwork::classify() {
  int which = -1, i = 0;
  float max = 0.00;
  for (auto& neuron:outputs->neurons) {
    float out = neuron->outputActivation;
    if (out >= max) {
      which = i;
      max = out;
    }
    i++;
  }
  return which;
}

void NeuralNetwork::print() {
  cout << "Inputs:\n";
  inputs->print();
  cout << "Hidden:\n";
  hidden->print();
  cout << "Outputs:\n";
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

float Trainer::meanSquaredError(vector<vector<float>> inps, vector<vector<float>> expected_) {
  float sum = 0, count = 0;
  for (auto& inp:inps) {
    int i = 0;
    int inNeuronNum = 0;
    for (auto& ninp:network->inputs->neurons)
      ninp->outputActivation = inp[inNeuronNum++];
    network->computeOutputs();
    auto expected = expected_[i++];
    for (size_t i=0; i<expected.size(); i++) {
      float diff = expected[i] - network->outputs->neurons[i]->outputActivation;
      sum += pow(diff,2);
    }
    count += expected.size();
  }
  return sum/count;
}


void Trainer::calcGradients(vector<vector<float>> inps, vector<vector<float>> desired_) {
  NeuralNetwork* net = network;

  //net->hidden->randomWeights();
  //net->outputs->randomWeights();

  // first reset all the weight derivative accumulators
  for (auto& neuron:net->outputs->neurons) {
    for (auto& input:neuron->inputs) {
      input->deriv = 0;
    }
  }
  for (auto& neuron:net->hidden->neurons) {
    for (auto& input:neuron->inputs) {
      input->deriv = 0;
    }
  }
  int in = 0;

  for (auto& desired:desired_) {
    float dEdY[1000], dEdX[1000];
    int inNeuronNum = 0;
    vector<float> exampleInputs = inps[in];
    for (auto& ninp:net->inputs->neurons)
      ninp->outputActivation = exampleInputs[inNeuronNum++];
    net->computeOutputs();

    int j = 0;

    std::vector<float>& d = desired;
    for (auto& neuron:net->outputs->neurons) {
      float Yj = neuron->outputActivation;
      dEdY[j] = Yj - d[j];
      dEdX[j] = dEdY[j] * Yj * (1 - Yj);
      for (auto& input:neuron->inputs) {
        float Yi = input->fromNeuron->outputActivation;
        float dEdWij = dEdX[j] * Yi;
        input->deriv += dEdWij;
      }
      j++;
    }
    int i = 0;
    
    for (auto& neuron:net->hidden->neurons) {
      float sum = 0;
      j = 0;
      int jj = 0;
      for (auto& out:net->outputs->neurons) {
        NeuralInput* matchInput = out->findInputFrom(neuron);
        float Wji = matchInput->weight;
        sum += dEdX[jj] * Wji;
        jj++;
      }

      dEdY[j] = sum;
      float Yj = neuron->outputActivation;
      dEdX[j] = dEdY[j] * Yj * (1 - Yj);
      for (auto& input:neuron->inputs) {
        float Yi = input->fromNeuron->outputActivation;
        float dEdWij = dEdX[j] * Yi;
        input->deriv += dEdWij;
      }
      i++;
    }
    in++;
  }

  for (auto& neuron:net->outputs->neurons) {
    for (auto& input:neuron->inputs) {
      float weightAdjust = -1 * learningRate * input->changespeed * input->deriv;
      input->weight += weightAdjust;
      //input->changespeed += input->deriv;      
      //cout << "outp adjust=" << weightAdjust << " weight=" << input->weight << "\n";
    }
  }
  for (auto& neuron:net->hidden->neurons) {
    for (auto& input:neuron->inputs) {
      float weightAdjust = -1 * learningRate * input->changespeed * input->deriv;
      input->weight += weightAdjust;
      //input->changespeed += input->deriv;
    }
  }
}




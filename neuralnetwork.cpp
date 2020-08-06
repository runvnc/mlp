#include <math.h>
#include "neuralnetwork.h"
#include <iostream>
#include <assert.h>
#include <cmath>

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
  assert(outputs);
  inputs->connectTo(hidden); 
  hidden->connectTo(outputs);
}

void NeuralNetwork::computeOutputs() {
  hidden->activateAll();
  outputs->activateAll();
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

float Trainer::meanSquaredError(std::vector<float> expected) {
  float sum = 0;
  for (size_t i=0; i<expected.size(); i++) {
    float diff = expected[i] - network->outputs->neurons[i]->outputActivation;
    sum += pow(diff,2);
  }
  return sum/expected.size();
}


void Trainer::calcGradients(std::vector<float> desired) {
  NeuralNetwork* net = network;
  // for each output, starting at last layer, working backward
  float dEdY[1000], dEdX[1000];
  int j = 0;

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

  //ss

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
/*
  adjust all of the weights

      float weightAdjust = -1 * learningRate * input->deriv (dEdWij);
      input->weight += weightAdjust;
*/

}

/*
inside of/ outside of
relation between two objects
describes categories of scenarios
inside if will be obscured/hidden if perspective changes
or will need to move in a certain way to get out
its an abstract description of geometry
so there needs to be some kind of partial or complete enclosure
but this can include cups or bowls that are not entirely enclosed


need to be able to quickly retrieve episodes by tagged concept
need high performance (parallel?) transformations involving many
inputs and outputs
and a way to automatically create those composed functions
to arrive at outputs that match inputs (modeling)

would be nice if that could be plugged in to a different robot
*/

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

"Episodic memory" comes the closest. I’ve repeatedly rejigged existing libraries to meet what's needed for predicate-based knowledge.

There aren't many papers these days that address unsupervised concept creation, much less predicates that connect them. When NNs deal with concepts, they are dictated by the trainer. This is why we have the “symbol grounding problem” - how do you ground the meaning of a word in reality?

The approach I take is to first create a new network for every learning episode. Each is a small predicate, like an isolated memory. Back propagation then prunes and groups them (unsupervised) into concepts, resulting in a network of these networks connected to one another.

The main innovation is to group concepts based on the agent’s motivations, rather than on data clusters. All networks that meet a particular motivation are part of a concept.

Grouping concepts based on the agent’s motivations implies that truth is subjective to each person. ML folk tend to be uncomfortable with the idea that truth is subjective, rather than data-driven. But I see no other path to solve the symbol grounding problem, because the meaning of something is unique to your personal motivations.

*/



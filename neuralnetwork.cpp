#include <math.h>

void Neuron::calculateActivation() {
  float sum = 0;

  for (auto& input : inputs) {
    sum += (input.weight * input.outputActivation) - bias;
  }
 
  sum *= -1;
  activation = 1.0 / (1.0 + exp(sum));  
}


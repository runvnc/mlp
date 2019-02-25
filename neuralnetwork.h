#include <vector>

using namespace std;

struct NueralInput {
  Neuron* fromNeuron;
  Neuron* toNeuron;
  float weight;  
}

class Neuron {
  public:

    int layer;
    int index;

    float bias;

    vector<NeuralInput> inputs; 

    float outputActivation;
    
    void calculateActivation();
}

class Layer {
  vector<Neuron> neurons;
  void activateAll();
}

class NeuralNetwork {
  public:
    Layer inputs;
    vector<Layer> hidden;
    Layer outputs;
}

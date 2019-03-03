#include <vector>

using namespace std;

class NueralInput {
  public:
    Neuron* fromNeuron;
    Neuron* toNeuron;
    float weight;

    NeuralInput(Neuron* from_, Neuron* to_, float weight_) {
      from = from_;
      to = to_;
      weight = weight_;
    }  
}

class Neuron {
  public:

    int layer;
    int index;

    float bias;

    vector<NeuralInput*> inputs; 

    float outputActivation;
    
    void calculateActivation();
}

class Layer {
  vector<Neuron*> neurons;
  void activateAll();
  connectTo(Layer*);
}

class NeuralNetwork {
  public:
    Layer* inputs;
    vector<Layer*> hidden;
    Layer* outputs;

    void connectLayers();
    randomizeHidden();
    void computeOutputs();
}

class MNISTNetwork: public NeuralNetwork {

}

class Trainer {
  public:
    NeuralNetwork* network;

    void compare(std::vector<float> expected);
    void train(std::vector<float> inputs, std::vector<float> outputs);

}



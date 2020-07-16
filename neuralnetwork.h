#include <vector>

using namespace std;

class Neuron;

class NeuralInput {
  public:
    Neuron* fromNeuron;
    Neuron* toNeuron;
    float weight = 0;

    NeuralInput(Neuron* from_, Neuron* to_, float weight_) {
      fromNeuron = from_;
      toNeuron = to_;
      weight = weight_;
    }  
};

class Neuron {
  public:

    int layer;
    int index;

    float bias = 0;

    vector<NeuralInput*> inputs; 

    float outputActivation;
    
    void calculateActivation();
};

class Layer {
  public:
    int ind = 0;
    vector<Neuron*> neurons;
    Layer(int, int);

    void activateAll();
    void connectTo(Layer*);
    void print();
};

class NeuralNetwork {
  public:
    Layer* inputs = NULL;
    vector<Layer*> hidden;
    Layer* outputs = NULL;

    void connectLayers();
    void randomizeHidden();
    void computeOutputs();
    void print();
};

class MNISTNetwork: public NeuralNetwork {

};



class Trainer {
  public:
    NeuralNetwork* network;
    Trainer(NeuralNetwork*);
    void compare(std::vector<float> expected);
    void train(std::vector<float> inputs, std::vector<float> outputs);

};



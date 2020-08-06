#include <vector>

using namespace std;

class Neuron;

class NeuralInput {
  public:
    Neuron* fromNeuron;
    Neuron* toNeuron;
    float weight = 0;
    float deriv = 0;

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
    NeuralInput* findInputFrom(Neuron*);
    
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
    Layer* hidden;
    Layer* outputs = NULL;

    void assignInput(size_t,float);
    void connectLayers();
    void randomizeHidden();
    void computeOutputs();
    void print();
};

class MNISTNetwork: public NeuralNetwork {

};

class Trainer {
  public:
    NeuralNetwork* network = 0;
    float learningRate = 0;
    Trainer(NeuralNetwork*, float rate);
    void compare(std::vector<float> expected);
    float meanSquaredError(std::vector<float> expected);
    void calcGradients(std::vector<float> expected);
};



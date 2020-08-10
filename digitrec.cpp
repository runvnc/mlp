#include "mnist.h"
#include <stdio.h>
#include "ANDnet.h"
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <string>


void setupMNIST() {
  int32_t TESTING_LABEL_MAGIC_NUM = 2049;
  auto testingLabels = new MNISTLabels(TESTING_LABEL_MAGIC_NUM);
  testingLabels->loadFile("digitsdata/t10k-labels-idx1-ubyte");

  printf("Loaded %d testing labels.\n", testingLabels->numItems);

  int32_t TESTING_IMAGES_MAGIC_NUM = 2051;
  auto testingImages = new MNISTImages(TESTING_IMAGES_MAGIC_NUM);
  testingImages->loadFile("digitsdata/t10k-images-idx3-ubyte");

  printf("Loaded %d testing images.\n", testingImages->numImages);

  NeuralNetwork* net = new NeuralNetwork();
      
  net->inputs = new Layer(0,784);
  net->hidden = new Layer(1,32);
  net->outputs = new Layer(2,10);
  cout << "Connect layers \n";
  net->connectLayers();
  cout << "OK\n";
  auto trainer = new Trainer(net, 0.005);
  int rows = 28;
  int cols = 28;
  vector<vector<float>> inputs,expected;
  vector<vector<float>> inputsx,expectedx;

  for (int index=0; index<230; index++) { 
    vector<float> inp, exp;
    char* pixels = testingImages->images[index];
    for (int row = 0; row < rows; row++) {
      for (int col = 0; col < cols; col++) {
        int index = row*cols+col;
        uint8_t darkness = pixels[index];
        inp.push_back(darkness/255.0);    
      }
    }
    if (index < 30) inputs.push_back(inp);
    inputsx.push_back(inp);
    for (int l=0;l<10;l++) exp.push_back(0);
    int which = testingLabels->labels[index];
    cout << "which = " << which << "\n";
    exp[which] = 1.0;
    if (index < 30) expected.push_back(exp); 
    expectedx.push_back(exp);
  }
  string d;
  getline(cin,d);
  float lowmse = 100;
  for (int k=0; k<100; k++) {
    cout << ">>> " << k << "\n";
    cout << "Randomizing weights\n";
    net->randomWeights();
    cout << "Calculating MSE\n";
    float mse = trainer->meanSquaredError(inputs, expected);
    cout << "Got MSE\n";
    if (mse < lowmse) {
      lowmse = mse;
      cout << "--------------------------------------------------\n";
      cout << "mse: " << mse << "\n";
      net->outputs->print();
      net->hidden->printWeights();
      net->outputs->printWeights();
    }
  }

  for (int i=0; i<1500; i++) {
    if (i%100==0) cout << "calc gradients: " << i << "\n";
    trainer->calcGradients(inputs, expected);
  } 

  cout << "Calculating MSE\n";
  float mse = trainer->meanSquaredError(inputs, expected);
  cout << "Got MSE\n";
  lowmse = mse;
  cout << "--------------------------------------------------\n";
  cout << "mse: " << mse << "\n";

  string xx;
  for (int t=200; t<214;t++) { 
    testingImages->printImage(t);
    net->assignInputs(inputsx[t]);
    net->computeOutputs();
    int c = net->classify();
    //net->print();
    cout << "Class = " << c << "\n";
    getline(cin, xx);
  }
}

int main(int argc, const char* argv[]) {
  srand (time(NULL));

  setupMNIST();
/*
  auto net = new ANDNetwork();

  auto trainer = new Trainer(net, 0.11);
  vector<vector<float>> exp;
  vector<float> exp_;
  exp_.push_back(0);
  exp_.push_back(1);
  exp.push_back(exp_);
  vector<float> exp2_;
  exp2_.push_back(1);
  exp2_.push_back(0);
  exp.push_back(exp2_);

  vector<float> exp3_;
  exp3_.push_back(1);
  exp3_.push_back(0);
  exp.push_back(exp3_);

  vector<vector<float>>inputs;
  vector<float> inp1;
  inp1.push_back(1);
  inp1.push_back(1);
  inputs.push_back(inp1);
  vector<float> inp2;
  inp2.push_back(0);
  inp2.push_back(1);
  inputs.push_back(inp2);

  vector<float> inp3;
  inp3.push_back(1);
  inp3.push_back(0);
  inputs.push_back(inp3);

  for (int nn=0; nn<5; nn++) {
    exp.push_back(exp_);
    inputs.push_back(inp1);
  }

  float lowmse = 100;
  for (int k=0; k<2000; k++) {
    net->randomWeights();
    float mse = trainer->meanSquaredError(inputs, exp);
    if (mse < lowmse) {
      lowmse = mse;
      cout << "--------------------------------------------------\n";
      cout << "mse: " << mse << "\n";
      net->outputs->print();
      net->hidden->printWeights();
      net->outputs->printWeights();
    }
  }

  for (int i=0; i<25000; i++) {
    trainer->calcGradients(inputs, exp);
  } 

  //cout << "Difference from expected:\n";
  //trainer->mean(exp);
  //net->assignInput(0,1);
  //net->assignInput(1,1);
  
  cout << "/////////////////////////////////////////////////\n";

  float mse = trainer->meanSquaredError(inputs, exp);
  cout << "mse: " << mse << "\n";

  net->assignInput(0,0);
  net->assignInput(1,0);
  net->computeOutputs();
  cout << "Network:\n";
  net->print();
  cout << "class: " << net->classify() << "\n"; */
}

/*
Standard knowledge encoding format
32x32 bytes = grid
as a type of language
high-level shapes described
as 32 x 32 
ways to compose multiple 32x32 grids
to describe larger or more complex shapes
mapping to images or visual fields
apply grids to eachother as mapping
standard ways to go from images to grid arrays
blowing up or expanding grids

*/
#include "mnist.h"
#include <stdio.h>
#include "ANDnet.h"
#include <iostream>
#include <stdlib.h>
#include <time.h>  


void setupMNIST() {


  int32_t TESTING_LABEL_MAGIC_NUM = 2049;
  auto testingLabels = new MNISTLabels(TESTING_LABEL_MAGIC_NUM);
  testingLabels->loadFile("digitsdata/t10k-labels-idx1-ubyte");

  printf("Loaded %d testing labels.\n", testingLabels->numItems);

  int32_t TESTING_IMAGES_MAGIC_NUM = 2051;
  auto testingImages = new MNISTImages(TESTING_IMAGES_MAGIC_NUM);
  testingImages->loadFile("digitsdata/t10k-images-idx3-ubyte");

  printf("Loaded %d testing images.\n", testingImages->numImages);

  for (int i=0; i < 5; i++) {
    testingImages->printImage(i);
    printf("%d\n",testingLabels->labels[i]);
  }
}

int main(int argc, const char* argv[]) {
  srand (time(NULL));

  auto net = new ANDNetwork();
  net->computeOutputs();
  cout << "Network:\n";
  net->print();
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
  //exp.push_back(exp3_);

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
  //inputs.push_back(inp3);

  for (int i=0; i<500000; i++) {
    trainer->calcGradients(inputs, exp);
  }
  //cout << "Difference from expected:\n";
  //trainer->mean(exp);
  //net->assignInput(0,1);
  //net->assignInput(1,1);

  net->assignInput(0,1);
  net->assignInput(1,1);
  net->computeOutputs();
  cout << "Network:\n";
  net->print();
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
#include "mnist.h"
#include <stdio.h>
#include "ANDnet.h"

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
  auto net = new ANDNetwork();
  net->computeOutputs();
  net->print();
  auto trainer = new Trainer(net);
  vector<float> exp;
  exp.push_back(0);
  exp.push_back(1);
  trainer->compare(exp);
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
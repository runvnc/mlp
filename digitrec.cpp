#include "mnist.h"
#include <stdio.h>

int main(int argc, const char* argv[]) {
  int32_t TESTING_MAGIC_NUM = 2049;
  auto testingLabels = new MNISTLabels(TESTING_MAGIC_NUM);
  testingLabels->loadFile("digitsdata/t10k-labels-idx1-ubyte");

  printf("Found %d testing labels.\n", testingLabels->numItems);
  for (int i=0; i<10; i++) {
    printf("%d\n", testingLabels->labels[i]);
  }

}

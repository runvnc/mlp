#pragma once

#include <string>
#include <vector>

using namespace std;


class MNISTLabels {
  public:
    MNISTLabels(int32_t);

    int32_t magicNumber;
    int32_t numItems;
    char* labels;

    void loadFile(string name);
};


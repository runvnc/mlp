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

class MNISTImages {
  public:
    MNISTImages(int32_t);

    int32_t magicNumber;
    int32_t numImages;
    int32_t rows;
    int32_t cols;

    vector<char *> images;

    void loadFile(string name);
    void printImage(int index);
};


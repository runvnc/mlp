#include <fstream>
#include "mnist.h"
#include <exception>
#include "textcolor.h"
#include <vector>

using namespace std;

struct BadMagic : public std::exception {
  const char * what() const throw() { return "Bad magic number"; } };

int32_t readSwap(fstream& file) {
  int32_t num;
  file.read((char*)&num, sizeof(num));
  return __builtin_bswap32(num);
}

MNISTLabels::MNISTLabels(int32_t magic) {
  magicNumber = magic;
}

void MNISTLabels::loadFile(string name) {
  fstream file(name.c_str(), ios::in|ios::binary);
  int32_t magic_ = readSwap(file);
  if (magic_ != magicNumber) throw BadMagic();
  numItems = readSwap(file); 
  labels = new char[numItems];
  file.read(labels, numItems);
  file.close();
}

MNISTImages::MNISTImages(int32_t magic) {
  magicNumber = magic;
}

void MNISTImages::loadFile(string name) {
  fstream file(name.c_str(), ios::in|ios::binary);
  int32_t magic_ = readSwap(file);
  if (magic_ != magicNumber) throw BadMagic();
  numImages = readSwap(file);
  rows = readSwap(file);
  cols = readSwap(file);
  
  for (int i = 0; i < numImages; i++) {
    char* pixels = new char[rows*cols];
    file.read(pixels, rows*cols);
    images.push_back(pixels);    
  }
  file.close();
}


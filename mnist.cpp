#include <fstream>
#include "mnist.h"
#include <exception>
#include "textcolor.h"

using namespace std;

struct BadMagic : public std::exception {
  const char * what() const throw() { return "Bad magic number"; } };

MNISTLabels::MNISTLabels(int32_t magic) {
  magicNumber = magic;
}

int32_t readSwap(fstream& file) {
  int32_t num;
  file.read((char*)&num, sizeof(num));
  return __builtin_bswap32(num);
}

void MNISTLabels::loadFile(string name) {
  fstream file(name.c_str(), ios::in|ios::binary);
  int32_t magic_ = readSwap(file);
  if (magic_ != magicNumber) {
    printf("Read magic: %d; expected: %d\n", magic_, magicNumber);
    throw BadMagic();
  }
  numItems = readSwap(file); 
  labels = new char[numItems];
  file.read(labels, numItems);
  file.close();
}


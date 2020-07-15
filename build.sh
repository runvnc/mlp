#!/bin/bash
#
g++ -std=c++17 -Wall -Wfatal-errors -fmax-errors=2 digitrec.cpp mnist.cpp neuralnetwork.cpp ANDNetwork.cpp -o digitrec

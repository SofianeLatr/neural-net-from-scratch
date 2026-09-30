#include <iostream>

struct typedef{

    int prevWidth;
    int width;

    float **weights;
    float *biases;
}Layer;

struct typedef{

    int hiddenLayers;
    Layer *layers;
}Network;
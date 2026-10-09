#include <iostream>
#include <random>

// Random number generator
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

typedef struct{

    int prevWidth;
    int width;

    float **weights;
    float *biases;
} Layer;

typedef struct{

    int hiddenLayers;
    Layer *layers;
    int maxWidth;
} Network;

void initNetwork(Network &net, int inputWidth, int hiddenLayers, int *hiddenWidths, int outputWidth)
{

    net.hiddenLayers = hiddenLayers;
    net.layers = new Layer[hiddenLayers + 1];

    for (int i = 0; i < hiddenLayers + 1; i++) {

        if (i == 0) {
            net.layers[i].prevWidth = inputWidth;
            net.layers[i].width = hiddenWidths[i];
        }
        else if (i == hiddenLayers) {
            net.layers[i].prevWidth = hiddenWidths[i - 1];
            net.layers[i].width = outputWidth;
        }
        else {
            net.layers[i].prevWidth = hiddenWidths[i - 1];
            net.layers[i].width = hiddenWidths[i];
        }

        net.layers[i].weights = new float*[net.layers[i].width];

        for (int j = 0; j < net.layers[i].width; j++) {

            net.layers[i].weights[j] =
                new float[net.layers[i].prevWidth];

            for (int k = 0; k < net.layers[i].prevWidth; k++)
                net.layers[i].weights[j][k] = dis(gen);
        }

        net.layers[i].biases =
            new float[net.layers[i].width];

        for (int j = 0; j < net.layers[i].width; j++)
            net.layers[i].biases[j] = dis(gen);
    }
    net.maxWidth = 0;
    
    for (int i = 0; i < net.hiddenLayers + 1; i++) {
        if (net.layers[i].width > net.maxWidth) {
            net.maxWidth = net.layers[i].width;
        }
    }
}

float relu(float x){
    return x > 0 ? x : 0;
}

float* forward(Network &net, float* input)
{
    float* output = new float[net.maxWidth];
    float* next = new float[net.maxWidth];

    float* prev = input;

    for (int i = 0; i < net.hiddenLayers + 1; i++) {

        for (int j = 0; j < net.layers[i].width; j++) {

            float sum = net.layers[i].biases[j];

            for (int k = 0; k < net.layers[i].prevWidth; k++)
                sum += net.layers[i].weights[j][k] * prev[k];

            next[j] = relu(sum);
        }

        float* temp = output;
        output = next;
        next = temp;

        prev = output;
    }

    delete[] next;

    return output;
}


void backPropagation(Network &net, float* input, float* target, float learningRate)
{
    int count = net.hiddenLayers + 1;

    float** activations = new float*[count + 1];

    activations[0] = input;

    for (int i = 0; i < count; i++) {
        Layer &layer = net.layers[i];

        activations[i + 1] = new float[layer.width];

        for (int j = 0; j < layer.width; j++) {
            float sum = layer.biases[j];

            for (int k = 0; k < layer.prevWidth; k++)
                sum += layer.weights[j][k] * activations[i][k];

            activations[i + 1][j] = sum > 0 ? sum : 0;
        }
    }

    float* error = new float[net.layers[count - 1].width];

    for (int j = 0; j < net.layers[count - 1].width; j++)
        error[j] = target[j] - activations[count][j];

    for (int i = count - 1; i >= 0; i--) {
        Layer &layer = net.layers[i];

        float* previousError = nullptr;

        if (i > 0) {
            previousError = new float[layer.prevWidth];

            for (int j = 0; j < layer.prevWidth; j++) {
                previousError[j] = 0;

                for (int k = 0; k < layer.width; k++)
                    previousError[j] += layer.weights[k][j] * error[k];
            }
        }

        for (int j = 0; j < layer.width; j++) {
            float gradient = error[j] * (activations[i + 1][j] > 0 ? 1 : 0);

            for (int k = 0; k < layer.prevWidth; k++)
                layer.weights[j][k] += learningRate * gradient * activations[i][k];

            layer.biases[j] += learningRate * gradient;
        }

        delete[] error;

        error = previousError;
    }

}
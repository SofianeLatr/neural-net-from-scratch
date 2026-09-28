#include <stdio.h>
#include <stdlib.h>

typedef struct {
    float w[4][3];
    float b[4];
} HiddenLayer;

typedef struct {
    float w[2][4];
    float b[2];
} OutputLayer;

typedef struct {
    HiddenLayer hidden;
    OutputLayer output;
} Network;

void initNetwork(Network *net)
{
    for (int i = 0; i < 4; i++) {
        net->b[i] = 0;

        for (int j = 0; j < 3; j++)
            net->hidden.w[i][j] =
                ((float)rand() / RAND_MAX) * 2 - 1;
    }

    for (int i = 0; i < 2; i++) {
        net->output.b[i] = 0;

        for (int j = 0; j < 4; j++)
            net->output.w[i][j] =
                ((float)rand() / RAND_MAX) * 2 - 1;
    }
}


float relu(float x)
{
    return x > 0 ? x : 0;
}

void forward(Network *net, float input[3], float output[2])
{
    float hidden[4];

    for (int i = 0; i < 4; i++) {

        hidden[i] = net->hidden.b[i];

        for (int j = 0; j < 3; j++)
            hidden[i] += net->hidden.w[i][j] * input[j];

        hidden[i] = relu(hidden[i]);
    }

    for (int i = 0; i < 2; i++) {

        output[i] = net->output.b[i];

        for (int j = 0; j < 4; j++)
            output[i] += net->output.w[i][j] * hidden[j];
    }
}


int main()
{
    Network net;

    initNetwork(&net);

    float input[3] = {1.0f, 2.0f, 3.0f};
    float output[2];

    forward(&net, input, output);

    printf("Output 1: %f\n", output[0]);
    printf("Output 2: %f\n", output[1]);

    return 0;
}

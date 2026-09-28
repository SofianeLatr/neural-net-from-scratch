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

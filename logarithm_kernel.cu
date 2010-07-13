#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cublas.h>

__global__ void logarithm(float *device_parts, float n, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx < N) {
        float x = 2*idx + 1;
        device_parts[idx] = (1/x)*pow( (n - 1)/(n + 1), x);
    }
}


int num_terms = 128;
size_t size_terms = num_terms*sizeof(int);

int main() {
   int i;
   float z = 34.7;
   float *device_parts;
   cudaMalloc((void**) &device_parts, size_terms);

   int ThreadsPerBlock = 16;
   int NumBlocks = (int) ((num_terms + ThreadsPerBlock - 1) / ThreadsPerBlock );
   logarithm <<< NumBlocks, ThreadsPerBlock, 1 >>> (device_parts, z, num_terms);
   float result = cublasSasum(num_terms, device_parts, 1);

   printf("%f \n", 2*result);

   cudaFree(device_parts);

   return 0;
}


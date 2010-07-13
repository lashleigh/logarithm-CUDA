#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>

__global__ void logarithm(float *device_parts, float n, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx < N) {
        float x = 2*idx + 1;
        device_parts[idx] = (1/x)*pow( (n - 1)/(n + 1), x);
    }
}

int num_terms = 32;
size_t size_terms = num_terms*sizeof(int);

int main() {
   int z = 16;
   float *device_parts;
   float *host_parts;
   cudaMalloc((void**) &device_parts, size_terms);
   cudaMallocHost((void**) &host_parts, size_terms);

   logarithm <<< 2, 16, 1 >>> (device_parts, z, num_terms);

   cudaMemcpy(host_parts, device_parts, size_terms, cudaMemcpyDeviceToHost);
   int i;
   float total = 0;
   for(i = 0; i < num_terms; i++) {
    printf("%d %f\n", i, host_parts[i]);
    total += host_parts[i];
   }
   printf("%f \n", 2*total);

   return 0;
}


#ifndef HELPER_CUDA_H
#define HELPER_CUDA_H

#include <cuda_runtime_api.h>
#include <cstdio>

void myCudaCall(cudaError_t cudaError, int line, char *file);

static __inline__ __device__ __host__ int divUp(int a, int b) 
{ 
  return (a % b != 0) ? (a / b + 1) : (a / b);
}

#endif//HELPER_CUDA_H

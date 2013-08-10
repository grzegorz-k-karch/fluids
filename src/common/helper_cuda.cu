#include "helper_cuda.h"

//==============================================================================
void myCudaCall(cudaError_t cudaError, int line, char *file)
{
  if (cudaError) {
    fprintf(stderr, "[CUDA] Error %d in file:\n%s: \nline %d: %s\n",
	    cudaError, file, line, cudaGetErrorString(cudaGetLastError()));
    exit(1);
  }
}
//==============================================================================

PROGRAM = compute

NVCC = nvcc
NVCCFLAGS += 
INCLUDE = -I../common/ -I/home/gkk/NVIDIA_CUDA-5.0_Samples/common/inc/ -I/usr/local/cuda/include/

all: compute.o compute_kernel.o

compute.o: compute.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) -c compute.cu
compute_kernel.o: compute_kernel.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) -c compute_kernel.cu

.PHONY : clean
clean:
	rm -f *.o *~

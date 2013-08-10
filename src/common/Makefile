NVCC = nvcc
NVCCFLAGS = 

helper_cuda.o: helper_cuda.cu helper_cuda.h
	$(NVCC) -I/usr/local/cuda/include/ -c $<

.PHONY : clean
clean:
	rm -f *.o *~

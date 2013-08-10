PROGRAM = fluids
INCDIRS = -I../compute -I../render -I../simulate -I/usr/local/cuda/include/ -I../common/

CXX = g++
CXXFLAGS += 

fluids.o: fluids.cpp 
	$(CXX) $(CXXFLAGS) $(INCDIRS) -c fluids.cpp

.PHONY : clean
clean:
	rm -f *.o *~

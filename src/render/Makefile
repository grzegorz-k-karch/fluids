CXX = g++
CXXFLAGS += 
LFLAGS    += -L/usr/lib -lGL -lGLU -lglut -ldl
INCLUDES  =  -I.  -IglslHelpers/ -I/usr/include -Itrackball/ -Igl3w/ -I../compute -I/usr/local/cuda/include/ -I../common/
OBJS = oglcontext.o render.o

all: $(OBJS)

oglcontext.o: oglcontext.cpp oglcontext.h
	$(CXX) $(CXXFLAGS) -c oglcontext.cpp $(INCLUDES)
render.o: render.cpp render.h
	$(CXX) $(CXXFLAGS) -c render.cpp $(INCLUDES)

.PHONY : clean
clean:
	rm -f *.o *~

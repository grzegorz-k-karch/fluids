PROGRAM = ../bin/fluids
OBJECTS = \
common/helper_cuda.o \
compute/compute.o \
compute/compute_kernel.o \
render/trackball/trackball.o \
render/glslHelpers/glslHelpers.o \
render/gl3w/gl3w.o \
render/oglcontext.o \
render/render.o \
fluids/fluids.o

NVCC = nvcc
LIBS = -lGL -lglut -ldl -lX11

CXXFLAGS = 
NVCCFLAGS = 


.PHONY: all
all: $(PROGRAM)

debug: export CXXFLAGS = -DDEBUG -g
debug: export NVCCFLAGS = -DDEBUG -g
debug: $(PROGRAM)

release: export CXXFLAGS = -O3
release: export NVCCFLAGS = -O3
release: $(PROGRAM)

$(PROGRAM): $(OBJECTS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LIBS)

common/%.o: common/%.cu common/%.h
	cd $(dir $@) && $(MAKE)
compute/%.o: compute/%.cu compute/compute.h
	cd $(dir $@) && $(MAKE)
render/trackball/%.o: render/trackball/%.cpp render/trackball/%.h
	cd $(dir $@) && $(MAKE)
render/glslHelpers/%.o: render/glslHelpers/%.cpp render/glslHelpers/%.h
	cd $(dir $@) && $(MAKE)
render/gl3w/%.o: render/gl3w/%.c render/gl3w/GL3/%.h
	cd $(dir $@) && $(MAKE)
render/%.o: render/%.cpp render/%.h
	cd $(dir $@) && $(MAKE)
render/%.o: render/%.cpp render/%.h
	cd $(dir $@) && $(MAKE)
fluids/%.o: fluids/%.cpp 
	cd $(dir $@) && $(MAKE)

SRCS = \
compute/compute.cu \
compute/compute.h \
compute/compute_kernel.cu \
render/trackball/trackball.cpp \
render/trackball/trackball.h \
render/glslHelpers/glslHelpers.cpp \
render/glslHelpers/glslHelpers.h \
render/gl3w/gl3w.c \
render/gl3w/GL3/gl3w.h \
render/oglcontext.cpp \
render/oglcontext.h \
render/render.cpp \
render/render.h \
fluids/fluids.cpp 

.PHONY: TAGS
TAGS:
	etags $(SRCS)

.PHONY : clean
clean:
	rm -f $(PROGRAM) *.o *~
	cd ./compute; make clean
	cd ./render; make clean
	cd ./render/gl3w; make clean
	cd ./render/glslHelpers; make clean
	cd ./render/trackball; make clean
	cd ./fluids; make clean

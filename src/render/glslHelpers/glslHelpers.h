#ifndef SHADERS_H
#define SHADERS_H

//#include <GL/glew.h>
#include "../gl3w/GL3/gl3.h"
#include "../gl3w/GL3/gl3w.h"
#include <GL/gl.h>
#include <GL/glext.h>

#define MAX_INFO_LOG_SIZE 2048

GLuint buildGLSLProgram(const GLchar** vertexShaderPath, int numVS,
			const GLchar** fragmentShaderPath, int numFS);

#endif//SHADERS_H

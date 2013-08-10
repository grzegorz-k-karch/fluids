#include "glslHelpers.h"
#include <stdlib.h>
#include <stdio.h>

GLchar* loadShaderText(const char* filename)
{
  GLchar* shaderText = NULL;
  GLint shaderLength = 0;
  FILE *fp;

  fp = fopen(filename, "r");
  if (fp != NULL)
    {
      while (fgetc(fp) != EOF)
	{
	  shaderLength++;
	}
      rewind(fp);
      // +1 for '\0' character that is needed in glShaderSource function
      // as we want it to read null-terminated source.
      shaderText = (GLchar*)malloc(shaderLength+1);
      if (shaderText != NULL)
	{
	  fread(shaderText, 1, shaderLength, fp);
	}
      shaderText[shaderLength] = '\0';
      fclose(fp);
    }
  else
    return NULL;

  return shaderText;
}

GLuint buildGLSLProgram(const GLchar** vertexShaderPath, int numVS,
			const GLchar** fragmentShaderPath, int numFS)
{
  GLuint vertexShader;
  GLuint fragmentShader;
  GLuint program;

  vertexShader = glCreateShader(GL_VERTEX_SHADER);
  fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
  program = glCreateProgram();
  GLint success;

  GLchar **vertexShaderText = (GLchar**)malloc(numVS*sizeof(GLchar*));
  GLchar **fragmentShaderText = (GLchar**)malloc(numFS*sizeof(GLchar*));

  for (int i = 0; i < numVS; i++)
      vertexShaderText[i] = loadShaderText(vertexShaderPath[i]);
  for (int i = 0; i < numFS; i++)
      fragmentShaderText[i] = loadShaderText(fragmentShaderPath[i]);

  glShaderSource(vertexShader, numVS, (const GLchar**)vertexShaderText, NULL);
  glShaderSource(fragmentShader, numFS, (const GLchar**)fragmentShaderText, NULL);

  // Compile vertex shader and check if succeeded
  glCompileShader(vertexShader);
  glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
  if (!success)
    {
      GLchar infoLog[MAX_INFO_LOG_SIZE];
      glGetShaderInfoLog(vertexShader, MAX_INFO_LOG_SIZE, NULL, infoLog);
      fprintf(stderr, "Error in vertex shader compilation.\nInfo log:");
      fprintf(stderr, "%s\n", infoLog);
      return 0;
    }
  // Compile fragment shader and check if succeeded
  glCompileShader(fragmentShader);
  glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
  if (!success)
    {
      GLchar infoLog[MAX_INFO_LOG_SIZE];
      glGetShaderInfoLog(fragmentShader, MAX_INFO_LOG_SIZE, NULL, infoLog);
      fprintf(stderr, "Error in fragment shader compilation.\nInfo log:");
      fprintf(stderr, "%s\n", infoLog);
      return 0;
    }

  // Attach shaders to program
  glAttachShader(program, vertexShader);
  glAttachShader(program, fragmentShader);

  // Link program and check if succeeded
  glLinkProgram(program);
  glGetProgramiv(program, GL_LINK_STATUS, &success);
  if (!success)
    {
      GLchar infoLog[MAX_INFO_LOG_SIZE];
      glGetProgramInfoLog(program, MAX_INFO_LOG_SIZE, NULL, infoLog);
      fprintf(stderr, "Error in program linkage.\nInfo log:");
      fprintf(stderr, "%s\n", infoLog);
      return 0;
    }
  // Validate program
  glGetProgramiv(program, GL_VALIDATE_STATUS, &success);
  if (!success)
    {
      GLchar infoLog[MAX_INFO_LOG_SIZE];
      glGetProgramInfoLog(program, MAX_INFO_LOG_SIZE, NULL, infoLog);
      fprintf(stderr, "Error in program validation.\nInfo log:");
      fprintf(stderr, "%s\n", infoLog);
      return 0;
    }

  glDeleteShader(vertexShader);
  glDeleteShader(fragmentShader);

  return program;
}

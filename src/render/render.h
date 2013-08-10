#ifndef RENDER_H
#define RENDER_H

#include "oglcontext.h"
#include "render.h"
#include "compute.h"
#include "data_structure.h"

#include <GL/gl.h>

class Render {

 public:

  // General methods
  int Init(int windowWidth, int windowHeight, dataInfo_t* dataInfo);
  void Terminate();
  void Run();
  GLuint GetVolumeTexture();
  void SetDataSource(Compute* compute);

 private:

  // General members
  GLint WindowWidth;
  GLint WindowHeight;
  OGLContext *Context;

  // Application-specific members
  GLint VolumeWidth;
  GLint VolumeHeight;
  GLint VolumeDepth;
  GLfloat VolumeRatio[3];
  GLuint ShaderProgram;
  GLuint QuadVArray;
  GLuint QuadBuffer;
  GLuint VolumeTexture;
  Compute *Fluid;

  GLfloat ModelTranslation[3];
  GLfloat ModelRotation[2];
  GLint ButtonState;
  GLint MousePosition[2];

  // General methods
  int InitContext();
  void InitGl3w();
  void InitGL();
  void Draw();
  void Transforms();

  // Callbacks
  int KeyboardCallback();
  void MouseButtonCallback();
  void MouseMotionCallback();
  void ResizeCallback();

  // Application-specific methods
  void LoadShaders();
  void DetachAndDelete(GLuint programObj);
  void InitTextures();
  void InitVolumeTexture();
  void InitGeometry();
  void SetTex3DParams(GLuint texture, GLint internalFormat, GLsizei width,
  		      GLsizei height, GLsizei depth, GLenum format, 
		      float* data);
};

#endif//RENDER_H

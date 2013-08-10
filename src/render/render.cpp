#include "gl3w/GL3/gl3w.h" // gl3w must be included before gl.h (in render.h)
#include "render.h"
#include "glslHelpers/glslHelpers.h"
#include "glm/glm.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "glm/gtc/matrix_transform.hpp"

#include <iostream>

//==============================================================================
int Render::Init(int windowWidth, int windowHeight, dataInfo_t* dataInfo)
{
  WindowWidth = windowWidth;
  WindowHeight = windowHeight;

  VolumeWidth = dataInfo->resolution[0];
  VolumeHeight = dataInfo->resolution[1];
  VolumeDepth = dataInfo->resolution[2];

  VolumeRatio[0] = (dataInfo->resolution[0])*(dataInfo->spacing[0]);
  VolumeRatio[1] = (dataInfo->resolution[1])*(dataInfo->spacing[1]);
  VolumeRatio[2] = (dataInfo->resolution[2])*(dataInfo->spacing[2]);

  int maxdim = VolumeRatio[0] > VolumeRatio[1] ? 0 : 1;
  maxdim = VolumeRatio[maxdim] > VolumeRatio[2] ? maxdim : 2;
  GLfloat maxsize = VolumeRatio[maxdim];
  VolumeRatio[0] /= maxsize;
  VolumeRatio[1] /= maxsize;
  VolumeRatio[2] /= maxsize;

  ModelTranslation[0] = 0.0f;
  ModelTranslation[1] = 0.0f;
  ModelTranslation[2] = -3.5f;
  ModelRotation[0] = 0.0f;
  ModelRotation[1] = 0.0f;
  MousePosition[0] = 0;
  MousePosition[1] = 0;
  ButtonState = 0;

  if (!InitContext()) {
    return 0;
  }
  InitGl3w();
  InitGL();

  return 1;
}
//==============================================================================
void Render::InitGl3w()
{
  if (gl3wInit()) {
    std::cerr << "Failed to initialize gl3w." << std::endl;
  }
}
//==============================================================================
int Render::InitContext()
{
  Context = new OGLContext;
  return Context->Init(WindowWidth, WindowHeight);
}
//==============================================================================
void Render::InitGL()
{
  glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
  glClearDepth(1.0f);
  glFrontFace(GL_CCW);

  LoadShaders();

  InitTextures();
  InitGeometry();
}
//==============================================================================
void Render::LoadShaders()
{
  DetachAndDelete(ShaderProgram);

  const GLchar *vertexShader[1] =
    {"/home/gkk/dev/fluids/src/render/shaders/v_basic.glsl"};
  const GLchar *fragmentShader[1] =
    {"/home/gkk/dev/fluids/src/render/shaders/f_volren.glsl"};

  ShaderProgram = buildGLSLProgram(vertexShader, 1, fragmentShader, 1);

  glBindAttribLocation(ShaderProgram, 0, "in_position");
  glBindFragDataLocation(ShaderProgram, 0, "out_color");

  glLinkProgram(ShaderProgram);
}
//==============================================================================
void Render::DetachAndDelete(GLuint programObj)
{
  if (!glIsProgram(programObj)) {
    return;
  }

  const GLsizei maxNumShaders = 1024;
  GLsizei numReturnedShaders = 0;
  GLuint shaders[maxNumShaders];

  glUseProgram(0);
  glGetAttachedShaders(programObj, maxNumShaders, &numReturnedShaders, shaders);

  for (GLsizei i = 0; i < numReturnedShaders; i++) {
    glDetachShader(programObj, shaders[i]);
    glDeleteShader(shaders[i]);
  }
  glDeleteProgram(programObj);
}
//==============================================================================
void Render::SetTex3DParams(GLuint texture, GLint internalFormat, GLsizei width,
			    GLsizei height, GLsizei depth, GLenum format,
			    float* data)
{
  glBindTexture(GL_TEXTURE_3D, texture);
  glTexImage3D(GL_TEXTURE_3D, 0, internalFormat, width, height, depth,
	       0, format, GL_FLOAT, data);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
}
//==============================================================================
void Render::InitVolumeTexture()
{
  glGenTextures(1, &VolumeTexture);
  SetTex3DParams(VolumeTexture, GL_R32F, VolumeWidth, VolumeHeight,
		 VolumeDepth, GL_RED, NULL);
}
//==============================================================================
void Render::InitTextures()
{
  InitVolumeTexture();
}
//==============================================================================
void Render::InitGeometry()
{
  glm::vec3 quad[4] = {
    glm::vec3(-1.0, -1.0,  0.0),
    glm::vec3(1.0, -1.0,  0.0),
    glm::vec3(1.0,  1.0,  0.0),
    glm::vec3(-1.0,  1.0,  0.0)
  };

  glGenVertexArrays(1, &QuadVArray);
  glBindVertexArray(QuadVArray);

  glGenBuffers(1, &QuadBuffer);
  glBindBuffer(GL_ARRAY_BUFFER, QuadBuffer);
  glEnableVertexAttribArray(0);
  glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3)*4, quad, GL_STATIC_DRAW);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);

  glBindVertexArray(0);
}
//==============================================================================
void Render::Run()
{
  int exitLoop = 0;

  while (!exitLoop) {

    Fluid->Update();

    Draw();

    // Callbacks
    MouseButtonCallback();
    MouseMotionCallback();
    ResizeCallback();
    exitLoop = KeyboardCallback();
  }
}
//==============================================================================
void Render::Transforms()
{
  const float fnear = 0.1f;
  const float ffar = 10.0f;
  const float fov = 45.0f;
  const float aspect = float(WindowWidth) / float(WindowHeight);
  glm::mat4 modelMX;

  modelMX = glm::translate(glm::mat4(1.0f), glm::vec3(ModelTranslation[0],
						      ModelTranslation[1],
						      ModelTranslation[2]));
  modelMX = glm::rotate(modelMX, ModelRotation[0], glm::vec3(1.0f, 0.0f, 0.0f));
  modelMX = glm::rotate(modelMX, ModelRotation[1], glm::vec3(0.0f, 1.0f, 0.0f));

  glm::mat4 modelInvTranspMX = glm::transpose(glm::inverse(modelMX));

  glUniformMatrix4fv(glGetUniformLocation(ShaderProgram, "modelInvTranspMat"),
		     1, GL_FALSE, glm::value_ptr(modelInvTranspMX));
}
//==============================================================================
void Render::Draw()
{
  glUseProgram(ShaderProgram);

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_3D, VolumeTexture);
  glUniform1i(glGetUniformLocation(ShaderProgram, "volumeTex"), 0);

  glUniform3fv(glGetUniformLocation(ShaderProgram, "ratio"),
	       1, VolumeRatio);

  glm::vec2 imageScale = glm::vec2(1.0/WindowWidth, 1.0/WindowHeight);
  glUniform2fv(glGetUniformLocation(ShaderProgram, "imageScale"),
	       1, glm::value_ptr(imageScale));

  Transforms();

  GLubyte indices[6] = {0,1,3,3,1,2};

  glBindVertexArray(QuadVArray);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_BYTE, indices);
  glBindVertexArray(0);

  Context->SwapBuffers();
}
//==============================================================================
int Render::KeyboardCallback()
{
  unsigned char key;
  int x;
  int y;
  int exitLoop = 0;

  if (Context->CheckKeyboard(key, x, y)) {
    switch (key) {
    case 27: // Escape
      exitLoop = 1;
    case 'r':
      LoadShaders();      
      break;
    default:
      break;
    }
  }
  return exitLoop;
}
//=============================================================================
void Render::MouseButtonCallback()
{
  int button;
  int x;
  int y;

  if (Context->CheckMouseButton(ButtonState, x, y)) {

    MousePosition[0] = x;
    MousePosition[1] = y;
  }
}
//=============================================================================
void Render::MouseMotionCallback()
{
  int x;
  int y;

  if (Context->CheckMouseMotion(x, y)) {

    float dx = float(x - MousePosition[0]);
    float dy = float(y - MousePosition[1]);

    if (ButtonState == 1) { // LMB

      ModelRotation[0] += dy/5.0f;
      ModelRotation[1] += dx/5.0f;
    }
    if (ButtonState == 4) { // LMB

      ModelTranslation[2] += dy*0.025f;
    }

    MousePosition[0] = x;
    MousePosition[1] = y;
  }
}
//=============================================================================
void Render::ResizeCallback()
{
  int x;
  int y;

  if (Context->CheckResizeRequest(x, y)) {

  }
}
//=============================================================================
void Render::Terminate()
{
  Context->Terminate();
}
//==============================================================================
GLuint Render::GetVolumeTexture()
{
  return VolumeTexture;
}
//==============================================================================
void Render::SetDataSource(Compute* compute)
{
  Fluid = compute;
}
//==============================================================================

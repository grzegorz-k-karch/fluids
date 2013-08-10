#ifndef OGLCONTEXT_H
#define OGLCONTEXT_H

#include <GL/glx.h>

class OGLContext {

 public:

  int Init(int windowWidth, int windowHeight);
  void Terminate();
  void SwapBuffers();
  bool CheckKeyboard(unsigned char& key, int& x, int& y);
  bool CheckMouseMotion(int& x, int& y);
  bool CheckMouseButton(int& button, int& x, int& y);
  bool CheckResizeRequest(int& x, int& y);

 private:

  bool IsGLXVersionSupported(int majorVersion, int minorVersion);
  GLXFBConfig ChooseBestFBConfig();
  XSetWindowAttributes SetWindowAttributes(const XVisualInfo *visualinfo);

  Display *Dpy;
  Window Win;
  GLXWindow GLXWin;
  GLXContext Ctx;

  int WindowWidth;
  int WindowHeight;
};

#endif//OGLCONTEXT

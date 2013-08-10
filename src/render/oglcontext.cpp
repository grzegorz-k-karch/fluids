#include "oglcontext.h"

#include <cstdlib> // getenv
#include <iostream>
#include <string>
#include <cstring>

#define GLX_CONTEXT_MAJOR_VERSION_ARB 0x2091
#define GLX_CONTEXT_MINOR_VERSION_ARB 0x2092

// a function pointer *glXCreateContextAttribsARBProc with arguments 
// (Display*, GLXFBConfig, GLXContext, Bool, const int*), returning GLXContext.
typedef GLXContext (*glXCreateContextAttribsARBProc)
(Display*, GLXFBConfig, GLXContext, Bool, const int*);

//==============================================================================
int OGLContext::Init(int windowWidth, int windowHeight)
{
  // Open display
  Dpy = XOpenDisplay(getenv("DISPLAY"));

  if (!IsGLXVersionSupported(1, 3)) {
    XCloseDisplay(Dpy);
    return 0;
  }

  GLXFBConfig fbconfig = ChooseBestFBConfig();
  XVisualInfo *visualinfo = glXGetVisualFromFBConfig(Dpy, fbconfig);
  XSetWindowAttributes setwindowattributes = SetWindowAttributes(visualinfo);
  unsigned long int valuemask = CWBorderPixel | CWColormap | CWEventMask;

  WindowWidth = windowWidth;
  WindowHeight = windowHeight;

  Win = XCreateWindow(Dpy, RootWindow(Dpy, visualinfo->screen), 0, 0, 
  		      WindowWidth, WindowHeight, 0,
  		      visualinfo->depth, InputOutput, visualinfo->visual,
  		      valuemask, &setwindowattributes);

  GLXWin = glXCreateWindow(Dpy, fbconfig, Win, NULL);

  // Map the window to the display
  XMapWindow(Dpy, Win);

  // share_list parameter should contain the other context when multiple 
  // contexts are used (from OpenGL Superbible, 4th Edition, p.724)
  // Ctx = glXCreateNewContext(Dpy, fbconfig, GLX_RGBA_TYPE, NULL, True); 
  // According to OpenGL SuperBible 5th Edition, glXCreateContextAttribsARB 
  // should be used instead og glXCreateNewContext to control the OpenGL 
  // version used.

  GLint attrib_list[] = {GLX_CONTEXT_MAJOR_VERSION_ARB, 3,
			 GLX_CONTEXT_MINOR_VERSION_ARB, 3,
			 0};

  // Useful link (also check wikipedia on 'typedef'):
  // http://stackoverflow.com/questions/4295432/typedef-function-pointer
  // glXCreateContextAttribsARB is the same function pointer as 
  // typedefed on top. Here the pointer to subroutine returned by 
  // glXGetProcAddressARB must by cast exactly as the given subroutine is 
  // defined in the gl.h or glx.h files (see IBM website).

  glXCreateContextAttribsARBProc glXCreateContextAttribsARB = 0;
  glXCreateContextAttribsARB = (glXCreateContextAttribsARBProc)
    glXGetProcAddressARB((const GLubyte*) "glXCreateContextAttribsARB");

  Ctx = glXCreateContextAttribsARB(Dpy, fbconfig, 0, True, attrib_list);

  // GLX version > 1.3 needed for glXMakeContextCurrent
  glXMakeContextCurrent(Dpy, GLXWin, GLXWin, Ctx);

  XFree(visualinfo);

  return 1;
}
//==============================================================================
void OGLContext::Terminate()
{
  glXMakeContextCurrent(Dpy, None, None, 0);
  glXDestroyContext(Dpy, Ctx);
  glXDestroyWindow(Dpy, GLXWin);
  XDestroyWindow(Dpy, Win);
  XCloseDisplay(Dpy);
}
//==============================================================================
GLXFBConfig OGLContext::ChooseBestFBConfig()
{
  // Get information on all the configs that meet the following criteria--------
  int nelements;
  int screen = DefaultScreen(Dpy);
  const int attrib_list[] = {GLX_X_RENDERABLE, True,
			     GLX_DRAWABLE_TYPE, GLX_WINDOW_BIT,
			     GLX_RENDER_TYPE, GLX_RGBA_BIT,
			     GLX_CONFIG_CAVEAT, GLX_NONE,
			     GLX_X_VISUAL_TYPE, GLX_TRUE_COLOR,
			     GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8,
			     GLX_BLUE_SIZE, 8, GLX_ALPHA_SIZE, 8,
			     GLX_DEPTH_SIZE, 24, GLX_STENCIL_SIZE, 8,
			     GLX_DOUBLEBUFFER, True, 0};
  GLXFBConfig *fbconfigs;
  fbconfigs = glXChooseFBConfig(Dpy, screen, attrib_list, &nelements);

  // Choose the best config-----------------------------------------------------
  GLXFBConfig bestfbconfig;
  int best_fbc = -1;
  int worst_fbc = -1;
  int best_num_samp = -1;
  int worst_num_samp = 999;

  for (int i = 0; i < nelements; i++) {

    XVisualInfo *vi = glXGetVisualFromFBConfig(Dpy, fbconfigs[i]);

    if (vi) {

      int samp_buf;
      int samples;

      glXGetFBConfigAttrib(Dpy, fbconfigs[i], GLX_SAMPLE_BUFFERS, &samp_buf);
      glXGetFBConfigAttrib(Dpy, fbconfigs[i], GLX_SAMPLES, &samples);

      if (best_fbc < 0 || samp_buf && samples > best_num_samp) {
	best_fbc = i;
	best_num_samp = samples;
      }
      if (worst_fbc < 0 || !samp_buf || samples < worst_num_samp) {
	worst_fbc = i;
      }
      worst_num_samp = samples;
    }
    XFree(vi);
  }
  bestfbconfig = fbconfigs[best_fbc];
  XFree(fbconfigs);

  return bestfbconfig;
}
//==============================================================================
XSetWindowAttributes OGLContext::SetWindowAttributes
(const XVisualInfo *visualinfo)
{
  XSetWindowAttributes swa;

  swa.colormap = XCreateColormap(Dpy, RootWindow(Dpy, visualinfo->screen),
				 visualinfo->visual, AllocNone);
  swa.background_pixmap = None;
  swa.border_pixel = 0;
  swa.event_mask = 
    StructureNotifyMask | 
    KeyPressMask | 
    ButtonPressMask |
    ButtonReleaseMask |
    PointerMotionMask |
    PointerMotionHintMask |
    VisibilityChangeMask | 
    ExposureMask |
    ResizeRedirectMask;

  return swa;
}
//==============================================================================
bool OGLContext::IsGLXVersionSupported(int majorVersion, int minorVersion)
{
  int queryMajor;
  int queryMinor;
  glXQueryVersion(Dpy, &queryMajor, &queryMinor);

  if (queryMajor == majorVersion && queryMinor < minorVersion) {

    std::cerr << "[OGLContext Error]: GLX 1.3 or greater is necessary" 
	      << std::endl;
    return false;
  }
  else {

    return true;
  }
}
//==============================================================================
void OGLContext::SwapBuffers()
{
  glXSwapBuffers(Dpy, GLXWin);
}
//==============================================================================
bool OGLContext::CheckKeyboard(unsigned char& key, int& x, int& y)
{
  XEvent xevent;

  if(XCheckWindowEvent(Dpy, Win, KeyPressMask, &xevent)) {

    char buffer[8];
    int bufSize = 8;
    KeySym keysym;
    XComposeStatus compose;
    int cnt = XLookupString(&xevent.xkey, buffer, bufSize, &keysym, &compose);

    if (cnt > 0) {

      key = buffer[0];
      x = xevent.xkey.x;
      y = xevent.xkey.y;

      return true;
    }
    else {
      return false;
    }
  }
  else {
    return false;
  }
}
//==============================================================================
bool OGLContext::CheckMouseMotion(int& x, int& y)
{
  XEvent xevent;

  if(XCheckWindowEvent(Dpy, Win, PointerMotionMask, &xevent)) {

    Window root_window, child_window;
    int root_x, root_y;
    unsigned int button_mask;

    XQueryPointer(Dpy, Win, &root_window, &child_window, 
		  &root_x, &root_y, &x, &y, &button_mask);
        
    return true;
  }
  else {
    return false;
  }
}
//==============================================================================
bool OGLContext::CheckMouseButton(int &button, int& x, int& y)
{
  XEvent xevent;

  if(XCheckWindowEvent(Dpy, Win, ButtonPressMask, &xevent)) {

    button |= 1<<(xevent.xbutton.button - 1);
    x = xevent.xmotion.x;
    y = xevent.xmotion.y;

    return true;
  }
  if(XCheckWindowEvent(Dpy, Win, ButtonReleaseMask, &xevent)) {

    button = 0;
    return true;
  }
  else {
    return false;
  }
}
//==============================================================================
bool OGLContext::CheckResizeRequest(int& x, int& y)
{
  XEvent xevent;

  if (XCheckWindowEvent(Dpy, Win, StructureNotifyMask, &xevent)) {

    // x = xevent.xconfigure.width;
    // y = xevent.xconfigure.height;

    // XResizeWindow(Dpy, Win, x, y);

    // std::cout << xevent.xresizerequest.width << " " 
    // 	      << xevent.xresizerequest.height << std::endl;

    return true;
  }
  return false;
}
//==============================================================================

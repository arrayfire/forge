/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

// Parts of this code sourced from SnopyDogy
// https://gist.github.com/SnopyDogy/a9a22497a893ec86aa3e

#include <fg/window.h>
#include <fg/exception.h>

#include <common.hpp>
#include <err_common.hpp>

namespace fg
{

static int gWindowCounter = 0;   // Window Counter

static void windowErrorCallback(int pError, const char* pDescription)
{
    fputs(pDescription, stderr);
}

#define GLFW_THROW_ERROR(msg, err) \
    throw fg::Error("Window constructor", __LINE__, msg, err);

Window::Window(int pWidth, int pHeight, const char* pTitle,
               const Window* pWindow, const bool invisible)
    : mWidth(pWidth), mHeight(pHeight), mWindow(NULL), mFont(NULL),
      mRows(0), mCols(0), mGLEWContext(NULL)
{
    CheckGL("Begin Window::Window");
    glfwSetErrorCallback(windowErrorCallback);

    if (!glfwInit()) {
        std::cerr << "ERROR: GLFW wasn't able to initalize\n";
        GLFW_THROW_ERROR("glfw initilization failed", fg::FG_ERR_GL_ERROR)
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    if (invisible)
        glfwWindowHint(GLFW_VISIBLE, GL_FALSE);
    else
        glfwWindowHint(GLFW_VISIBLE, GL_TRUE);
    // create glfw window
    // if pWindow is not null, then the window created in this
    // constructor call will share the context with pWindow
    GLFWwindow* temp = glfwCreateWindow(pWidth, pHeight, pTitle, NULL,
                                        (pWindow!=NULL ? pWindow->window() : NULL));
    if (temp == NULL) {
        std::cerr<<"Error: Could not Create GLFW Window!\n";
        GLFW_THROW_ERROR("glfw window creation failed", fg::FG_ERR_GL_ERROR)
    }
    mWindow = temp;

    // create glew context so that it will bind itself to windows
    if (pWindow==NULL) {
    GLEWContext* tmp = new GLEWContext();
    if (tmp == NULL) {
        std::cerr<<"Error: Could not create GLEW Context!\n";
        glfwDestroyWindow(mWindow);
        GLFW_THROW_ERROR("GLEW context creation failed", fg::FG_ERR_GL_ERROR)
    }
    mGLEWContext = tmp;
    } else {
        mGLEWContext = pWindow->glewContext();
    }

    // Set context (before glewInit())
    MakeContextCurrent(this);
    CheckGL("After MakeContextCurrent");

    //GLEW Initialization - Must be done
    glewExperimental = GL_TRUE;
    GLenum err = glewInit();
    if (err != GLEW_OK) {
        char buffer[128];
        sprintf(buffer, "GLEW init failed: Error: %s\n", glewGetErrorString(err));
        glfwDestroyWindow(mWindow);
        delete mGLEWContext;
        GLFW_THROW_ERROR(buffer, fg::FG_ERR_GL_ERROR);
    }
    err = glGetError();
    if (err!=GL_NO_ERROR && err!=GL_INVALID_ENUM) {
        // ignore this error, as GLEW is known to
        // have this issue with 3.2+ core profiles
        // they are yet to fix this in GLEW
        ForceCheckGL("GLEW initilization failed");
        GLFW_THROW_ERROR("GLEW initilization failed", fg::FG_ERR_GL_ERROR)
    }

    glfwSetWindowUserPointer(mWindow, this);

    auto keyboardCallback = [](GLFWwindow* w, int a, int b, int c, int d)
    {
        static_cast<Window*>(glfwGetWindowUserPointer(w))->keyboardHandler(a, b, c, d);
    };

    glfwSetKeyCallback(mWindow, keyboardCallback);
#ifdef WINDOWS_OS
    mCxt = glfwGetWGLContext(mWindow);
    mDsp = GetDC(glfwGetWin32Window(mWindow));
#endif
#ifdef LINUX_OS
    mCxt = glfwGetGLXContext(mWindow);
    mDsp = glfwGetX11Display();
#endif
    mID = gWindowCounter++; //set ID and Increment Counter!
    CheckGL("End Window::Window");
}

Window::~Window()
{
    MakeContextCurrent(this);
    if (mWindow!=NULL) glfwDestroyWindow(mWindow);
}

void Window::keyboardHandler(int pKey, int scancode, int pAction, int pMods)
{
    if (pKey == GLFW_KEY_ESCAPE && pAction == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(mWindow, GL_TRUE);
    }
}

void Window::draw(const Image& pImage)
{
    CheckGL("Begin drawImage");
    MakeContextCurrent(this);

    int wind_width, wind_height;
    glfwGetWindowSize(window(), &wind_width, &wind_height);
    glViewport(0, 0, wind_width, wind_height);

    // clear color and depth buffers
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(GRAY[0], GRAY[1], GRAY[2], GRAY[3]);

    pImage.render();

    glfwSwapBuffers(window());
    glfwPollEvents();
    CheckGL("End drawImage");
}

void Window::draw(const Plot& pHandle)
{
    CheckGL("Begin draw(Plot)");
    MakeContextCurrent(this);

    int wind_width, wind_height;
    glfwGetWindowSize(window(), &wind_width, &wind_height);
    glViewport(0, 0, wind_width, wind_height);

    glClearColor(GRAY[0], GRAY[1], GRAY[2], GRAY[3]);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    pHandle.render(0, 0, wind_width, wind_height);

    glfwSwapBuffers(window());
    glfwPollEvents();
    CheckGL("End draw(Plot)");
}

void Window::draw(const Histogram& pHist)
{
    CheckGL("Begin draw(Histogram)");
    MakeContextCurrent(this);

    int wind_width, wind_height;
    glfwGetWindowSize(window(), &wind_width, &wind_height);
    glViewport(0, 0, wind_width, wind_height);
    glClearColor(GRAY[0], GRAY[1], GRAY[2], GRAY[3]);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    pHist.render(0, 0, wind_width, wind_height);

    glfwSwapBuffers(window());
    glfwPollEvents();
    CheckGL("End draw(Histogram)");
}

void Window::grid(int pRows, int pCols)
{
    mRows = pRows;
    mCols = pCols;

    int wind_width, wind_height;

    MakeContextCurrent(this);
    glfwGetWindowSize(window(), &wind_width, &wind_height);
    glViewport(0, 0, wind_width, wind_height);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    mCellWidth = wind_width/ mCols;
    mCellHeight = wind_height/ mRows;
}

void Window::draw(int pColId, int pRowId,
                  const void* pRenderablePtr, Renderable pType,
                  const char* pTitle)
{
    static const float RED[4] = {1.0, 0.0, 0.0, 1.0};
    float pos[2] = {0.0, 0.0};

    CheckGL("Begin show(column, row)");
    int c     = pColId;
    int r     = pRowId;
    int x_off = c * mCellWidth;
    int y_off = (mRows-1-r) * mCellHeight;

    // set viewport to render sub image
    glViewport(x_off, y_off, mCellWidth, mCellHeight);
    glScissor(x_off, y_off, mCellWidth, mCellHeight);
    glEnable(GL_SCISSOR_TEST);
    glClearColor(GRAY[0], GRAY[1], GRAY[2], GRAY[3]);
    /* FIXME as of now, only fg::Image::render doesn't ask
     * for any input parameters */
    switch(pType) {
        case FG_IMAGE:
            ((const fg::Image*)pRenderablePtr)->render();
            break;
        case FG_PLOT:
            ((const fg::Plot*)pRenderablePtr)->render(x_off, y_off, mCellWidth, mCellHeight);
            break;
        case FG_HIST:
            ((const fg::Histogram*)pRenderablePtr)->render(x_off, y_off, mCellWidth, mCellHeight);
            break;
    }
    glDisable(GL_SCISSOR_TEST);
    mFont->setOthro2D(mCellWidth, mCellHeight);
    pos[0] = mCellWidth/3.0f;
    pos[1] = mCellHeight*0.9f;
    mFont->render(pos, RED, pTitle, 16);
    CheckGL("End show(column, row)");
}

void Window::draw()
{
    CheckGL("Begin show");
    glfwSwapBuffers(window());
    glfwPollEvents();
    CheckGL("End show");
}

void makeCurrent(Window* pWindow)
{
    MakeContextCurrent(pWindow);
}

}

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

static void keyboardCallback(GLFWwindow* pWind, int pKey, int scancode, int pAction, int pMods)
{
    if (pKey == GLFW_KEY_ESCAPE && pAction == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(pWind, GL_TRUE);
    }
}

#define GLFW_THROW_ERROR(msg, err) \
    throw fg::Error("Window constructor", __LINE__, msg, err);

Window::Window(int pWidth, int pHeight, const char* pTitle, const Window* pWindow)
    : mWidth(pWidth), mHeight(pHeight)
{
    CheckGL("Begin Window::Window");

    glfwSetErrorCallback(windowErrorCallback);

    if (!glfwInit()) {
        std::cerr << "ERROR: GLFW wasn't able to initalize\n";
        GLFW_THROW_ERROR("glfw initilization failed", fg::FG_ERR_GL_ERROR)
    }

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

    //GLEW Initialization - Must be done
    GLenum err = glewInit();
    if (err != GLEW_OK) {
        char buffer[128];
        sprintf(buffer, "GLEW init failed: Error: %s\n", glewGetErrorString(err));
        glfwDestroyWindow(mWindow);
        delete mGLEWContext;
        GLFW_THROW_ERROR(buffer, fg::FG_ERR_GL_ERROR);
    }

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

void makeWindowCurrent(Window* pWindow)
{
    MakeContextCurrent(pWindow);
}

}

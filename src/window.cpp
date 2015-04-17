/*******************************************************
 * Copyright (c) 2014, ArrayFire
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

static Window* gPrimaryWindow;  // This window is the primary window in forge, all subsequent
                                // windows use the same OpenGL context that is created while
                                // creating the primary window

static int gWindowCounter = 0;   // Window Counter

static bool isPrimaryWindowNotInitialized = true;

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

static void initializePrimaryWindow()
{
    // Create the Primary Window if not already done
    if (isPrimaryWindowNotInitialized) {
        // Initalize GLFW
        glfwSetErrorCallback(windowErrorCallback);
        if (!glfwInit()) {
            std::cerr << "ERROR: GLFW wasn't able to initalize" << std::endl;
            throw fg::Error("initializePrimaryWindow", __LINE__,
                    "glfw initilization failed", fg::FG_ERR_GL_ERROR);
        }
        // try creating the primary window
        GLFWwindow* temp = glfwCreateWindow(16, 16, "Alpha", NULL, NULL);
        if (temp == NULL) {
            printf("Error: Could not Create GLFW Window!\n");
            throw fg::Error("initializePrimaryWindow", __LINE__,
                    "glfw window creation failed", fg::FG_ERR_GL_ERROR);
        }
        // create glew context so that it will bind itself to windows
        GLEWContext* cxt = new GLEWContext();
        if (cxt == NULL)
        {
            printf("Error: Could not create GLEW Context!\n");
            glfwDestroyWindow(temp);
            throw fg::Error("initializePrimaryWindow", __LINE__,
                    "GLEW context creation failed", fg::FG_ERR_GL_ERROR);
        }

        gPrimaryWindow = new Window(16, 16, "Alpha");
        fg::Window::setGLEWcontext(cxt);

        // Set context (before glewInit())
        MakeContextCurrent(gPrimaryWindow);

        //GLEW Initialization - Must be done
        GLenum err = glewInit();
        if (err != GLEW_OK) {
            char buffer[128];
            sprintf(buffer, "GLEW init failed: Error: %s\n", glewGetErrorString(err));
            glfwDestroyWindow(gPrimaryWindow->window());
            delete gPrimaryWindow;
            throw fg::Error("initializePrimaryWindow", __LINE__, buffer, fg::FG_ERR_GL_ERROR);
        }

        isPrimaryWindowNotInitialized = false;
    }
}

Window::Window()
{
    initializePrimaryWindow();
}

Window::Window(int pWidth, int pHeight, const char* pTitle)
    : mWidth(pWidth), mHeight(pHeight)
{
    CheckGL("Begin Window::Window");
    initializePrimaryWindow();

    mWindow = glfwCreateWindow(pWidth, pHeight, pTitle, NULL, (GLFWwindow*)gPrimaryWindow->window());
    if (mWindow == NULL) {
        std::cerr<<"Error: Could not Create GLFW Window!\n";
        throw fg::Error("initializePrimaryWindow", __LINE__,
                "glfw window creation failed", FG_ERR_GL_ERROR);
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
    glfwDestroyWindow(mWindow);
}

const ContextHandle Window::context() const { return mCxt; }

const DisplayHandle Window::display() const { return mDsp; }

int Window::width() const { return mWidth; }

int Window::height() const { return mHeight; }

GLFWwindow* Window::window() const { return mWindow; }

void makeWindowCurrent(Window* pWindow)
{
    MakeContextCurrent(pWindow);
}

}

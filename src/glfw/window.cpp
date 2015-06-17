/*******************************************************
* Copyright (c) 2015-2019, ArrayFire
* All rights reserved.
*
* This file is distributed under 3-clause BSD license.
* The complete license agreement can be obtained at:
* http://arrayfire.com/licenses/BSD-3-Clause
********************************************************/

#include <common.hpp>
#include <glfw/window.hpp>

#include <iostream>

#define GLFW_THROW_ERROR(msg, err) \
    throw fg::Error("Window constructor", __LINE__, msg, err);

namespace wtk
{

Widget::Widget()
    : mWindow(NULL)
{
}

Widget::Widget(int pWidth, int pHeight, const char* pTitle, const Widget* pWindow, const bool invisible)
{
    if (!glfwInit()) {
        std::cerr << "ERROR: GLFW wasn't able to initalize\n";
        GLFW_THROW_ERROR("glfw initilization failed", fg::FG_ERR_GL_ERROR)
    }

    auto wndErrCallback = [](int errCode, const char* pDescription)
    {
        fputs(pDescription, stderr);
    };
    glfwSetErrorCallback(wndErrCallback);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    if (invisible)
        glfwWindowHint(GLFW_VISIBLE, GL_FALSE);
    else
        glfwWindowHint(GLFW_VISIBLE, GL_TRUE);

    glfwWindowHint(GLFW_SAMPLES, 4);
    mWindow = glfwCreateWindow(pWidth, pHeight, pTitle, nullptr,
                               (pWindow!=nullptr ? pWindow->getNativeHandle(): nullptr));

    if (!mWindow) {
        std::cerr<<"Error: Could not Create GLFW Window!\n";
        GLFW_THROW_ERROR("glfw window creation failed", fg::FG_ERR_GL_ERROR)
    }

    glfwSetWindowUserPointer(mWindow, this);
    auto kbCallback = [](GLFWwindow* w, int pKey, int pScancode, int pAction, int pMods)
    {
        static_cast<Widget*>(glfwGetWindowUserPointer(w))->keyboardHandler(pKey, pScancode, pAction, pMods);
    };
    glfwSetKeyCallback(mWindow, kbCallback);
}

Widget::~Widget()
{
    if (mWindow)
        glfwDestroyWindow(mWindow);
}

GLFWwindow* Widget::getNativeHandle() const
{
    return mWindow;
}

void Widget::makeContextCurrent() const
{
    glfwMakeContextCurrent(mWindow);
}

long long Widget::getGLContextHandle()
{
#ifdef OS_WIN
    return reinterpret_cast<long long>(glfwGetWGLContext(mWindow));
#endif
#ifdef OS_LNX
    return reinterpret_cast<long long>(glfwGetGLXContext(mWindow));
#endif
}

long long Widget::getDisplayHandle()
{
#ifdef OS_WIN
    return reinterpret_cast<long long>(GetDC(glfwGetWin32Window(mWindow)));
#endif
#ifdef OS_LNX
    return reinterpret_cast<long long>(glfwGetX11Display());
#endif
}

void Widget::getFrameBufferSize(int* pW, int* pH)
{
    glfwGetFramebufferSize(mWindow, pW, pH);
}

void Widget::setTitle(const char* pTitle)
{
    glfwSetWindowTitle(mWindow, pTitle);
}

void Widget::setPos(int pX, int pY)
{
    glfwSetWindowPos(mWindow, pX, pY);
}

void Widget::swapBuffers()
{
    glfwSwapBuffers(mWindow);
}

void Widget::hide()
{
    glfwHideWindow(mWindow);
}

void Widget::show()
{
    glfwShowWindow(mWindow);
}

bool Widget::close()
{
    return glfwWindowShouldClose(mWindow) != 0;
}

void Widget::keyboardHandler(int pKey, int pScancode, int pAction, int pMods)
{
    if (pKey == GLFW_KEY_ESCAPE && pAction == GLFW_PRESS) {
        glfwSetWindowShouldClose(mWindow, GL_TRUE);
    }
}

void Widget::pollEvents()
{
    glfwPollEvents();
}

}

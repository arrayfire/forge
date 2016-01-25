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

#include <glm/gtc/matrix_transform.hpp>

#include <iostream>

using glm::rotate;
using glm::translate;
using glm::scale;

#define GLFW_THROW_ERROR(msg, err) \
    throw fg::Error("Window constructor", __LINE__, msg, err);

namespace wtk
{

Widget::Widget()
    : mWindow(NULL), mClose(false), mLastXPos(0), mLastYPos(0), mMVP(glm::mat4(1.0f)), mButton(-1)
{
}

Widget::Widget(int pWidth, int pHeight, const char* pTitle, const Widget* pWindow, const bool invisible)
    : mClose(false), mLastXPos(0), mLastYPos(0), mMVP(glm::mat4(1.0f)), mButton(-1)
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

    auto closeCallback = [](GLFWwindow* w)
    {
        static_cast<Widget*>(glfwGetWindowUserPointer(w))->hide();
    };

    auto cursorCallback = [](GLFWwindow* w, double xpos, double ypos)
    {
        static_cast<Widget*>(glfwGetWindowUserPointer(w))->cursorHandler(xpos, ypos);
    };

    auto mouseButtonCallback = [](GLFWwindow* w, int button, int action, int mods)
    {
        static_cast<Widget*>(glfwGetWindowUserPointer(w))->mouseButtonHandler(button, action, mods);
    };

    glfwSetWindowCloseCallback(mWindow, closeCallback);
    glfwSetKeyCallback(mWindow, kbCallback);
    glfwSetCursorPosCallback(mWindow, cursorCallback);
    glfwSetMouseButtonCallback(mWindow, mouseButtonCallback);
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
#elif OS_LNX
    return reinterpret_cast<long long>(glfwGetGLXContext(mWindow));
#else
    return 0;
#endif
}

long long Widget::getDisplayHandle()
{
#ifdef OS_WIN
    return reinterpret_cast<long long>(GetDC(glfwGetWin32Window(mWindow)));
#elif OS_LNX
    return reinterpret_cast<long long>(glfwGetX11Display());
#else
    return 0;
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

void Widget::setSize(unsigned pW, unsigned pH)
{
    glfwSetWindowSize(mWindow, pW, pH);
}

void Widget::swapBuffers()
{
    glfwSwapBuffers(mWindow);
}

void Widget::hide()
{
    mClose = true;
    glfwHideWindow(mWindow);
}

void Widget::show()
{
    mClose = false;
    glfwShowWindow(mWindow);
}

bool Widget::close()
{
    return mClose;
}

void Widget::resetCloseFlag()
{
    if(mClose==true) {
        show();
    }
}

void Widget::keyboardHandler(int pKey, int pScancode, int pAction, int pMods)
{
    if (pKey == GLFW_KEY_ESCAPE && pAction == GLFW_PRESS) {
        hide();
    }
}

void Widget::cursorHandler(const float pXPos, const float pYPos)
{
    static const float SPEED = 0.005f;

    float deltaX = mLastXPos - pXPos;
    float deltaY = mLastYPos - pYPos;

    if (mButton == GLFW_MOUSE_BUTTON_LEFT) {
        // Translate
        mMVP = translate(mMVP, glm::vec3(-deltaX, deltaY, 0.0f) * SPEED);

    } else if (mButton == GLFW_MOUSE_BUTTON_LEFT + 10 * GLFW_MOD_ALT ||
               mButton == GLFW_MOUSE_BUTTON_LEFT + 10 * GLFW_MOD_CONTROL) {
        // Zoom
        if(deltaY != 0) {
            if(deltaY < 0) {
                deltaY = 1.0 / (-deltaY);
            }
            mMVP = scale(mMVP, glm::vec3(pow(deltaY, SPEED)));
        }
    } else if (mButton == GLFW_MOUSE_BUTTON_RIGHT) {
        int width, height;
        glfwGetWindowSize(mWindow, &width, &height);

        glm::vec3 curPos = trackballPoint(pXPos, pYPos, width, height);
        glm::vec3 delta = mLastPos - curPos;
        float angle = glm::radians(90.0f * sqrt(delta.x*delta.x + delta.y*delta.y + delta.z*delta.z));
        glm::vec3 axis(
                mLastPos.y*curPos.z-mLastPos.z*curPos.y,
                mLastPos.z*curPos.x-mLastPos.x*curPos.z,
                mLastPos.x*curPos.y-mLastPos.y*curPos.x
                );
        float dMag = sqrt(dot(delta, delta));
        float aMag = sqrt(dot(axis, axis));
        if (dMag>0 && aMag>0) {
            mMVP = rotate(mMVP, angle, axis);
        }
        mLastPos  = curPos;
    }

    mLastXPos = pXPos;
    mLastYPos = pYPos;
}

void Widget::mouseButtonHandler(int pButton, int pAction, int pMods)
{
    mButton = -1;
    if (pButton == GLFW_MOUSE_BUTTON_LEFT && pAction == GLFW_PRESS) {
        mButton = GLFW_MOUSE_BUTTON_LEFT;
    } else if (pButton == GLFW_MOUSE_BUTTON_RIGHT && pAction == GLFW_PRESS) {
        mButton = GLFW_MOUSE_BUTTON_RIGHT;
    } else if (pButton == GLFW_MOUSE_BUTTON_MIDDLE && pAction == GLFW_PRESS) {
        mButton = GLFW_MOUSE_BUTTON_MIDDLE;
    }
    if(pMods == GLFW_MOD_ALT || pMods == GLFW_MOD_CONTROL) {
        mButton += 10 * pMods;
    }
    // reset UI transforms upon mouse middle click
    if(pButton == GLFW_MOUSE_BUTTON_MIDDLE && pMods == GLFW_MOD_CONTROL && pAction == GLFW_PRESS) {
        mMVP = glm::mat4(1.0f);
    }
}

void Widget::pollEvents()
{
    glfwPollEvents();
}

}

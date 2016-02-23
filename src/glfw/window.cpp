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
    : mWindow(NULL), mClose(false), mLastXPos(0), mLastYPos(0), mButton(-1),
   mWidth(512), mHeight(512), mRows(1), mCols(1)
{
    mCellWidth  = mWidth;
    mCellHeight = mHeight;
    mFramePBO   = 0;
}

Widget::Widget(int pWidth, int pHeight, const char* pTitle, const Widget* pWindow, const bool invisible)
    : mWindow(NULL), mClose(false), mLastXPos(0), mLastYPos(0), mButton(-1), mRows(1), mCols(1)
{
    mFramePBO   = 0;

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

    auto rsCallback = [](GLFWwindow* w, int pWidth, int pHeight)
    {
        static_cast<Widget*>(glfwGetWindowUserPointer(w))->resizeHandler(pWidth, pHeight);
    };

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

    glfwSetFramebufferSizeCallback(mWindow, rsCallback);
    glfwSetWindowCloseCallback(mWindow, closeCallback);
    glfwSetKeyCallback(mWindow, kbCallback);
    glfwSetCursorPosCallback(mWindow, cursorCallback);
    glfwSetMouseButtonCallback(mWindow, mouseButtonCallback);

    glfwGetFramebufferSize(mWindow, &mWidth, &mHeight);
    mCellWidth  = mWidth;
    mCellHeight = mHeight;
}

Widget::~Widget()
{
    glDeleteBuffers(1, &mFramePBO);

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

    glReadBuffer(GL_FRONT);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, mFramePBO);
    glReadPixels(0, 0, mWidth, mHeight, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
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

void Widget::resizeHandler(int pWidth, int pHeight)
{
    mWidth      = pWidth;
    mHeight     = pHeight;
    mCellWidth  = mWidth  / mCols;
    mCellHeight = mHeight / mRows;
    resizePixelBuffers();
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

    int r, c;
    getViewIds(&r, &c);
    glm::mat4& mvp = mMVPs[r+c*mRows];

    if (mButton == GLFW_MOUSE_BUTTON_LEFT) {
        // Translate
        mvp = translate(mvp, glm::vec3(-deltaX, deltaY, 0.0f) * SPEED);

    } else if (mButton == GLFW_MOUSE_BUTTON_LEFT + 10 * GLFW_MOD_ALT ||
               mButton == GLFW_MOUSE_BUTTON_LEFT + 10 * GLFW_MOD_CONTROL) {
        // Zoom
        if(deltaY != 0) {
            if(deltaY < 0) {
                deltaY = 1.0 / (-deltaY);
            }
            mvp = scale(mvp, glm::vec3(pow(deltaY, SPEED)));
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
            mvp = rotate(mvp, angle, axis);
        }
        mLastPos  = curPos;
    }

    mLastXPos = pXPos;
    mLastYPos = pYPos;
}

void Widget::mouseButtonHandler(int pButton, int pAction, int pMods)
{
    mButton = -1;
    if (pAction == GLFW_PRESS) {
        switch(pButton) {
            case GLFW_MOUSE_BUTTON_LEFT  : mButton = GLFW_MOUSE_BUTTON_LEFT  ; break;
            case GLFW_MOUSE_BUTTON_RIGHT : mButton = GLFW_MOUSE_BUTTON_RIGHT ; break;
            case GLFW_MOUSE_BUTTON_MIDDLE: mButton = GLFW_MOUSE_BUTTON_MIDDLE; break;
        }
    }
    if (pMods == GLFW_MOD_ALT || pMods == GLFW_MOD_CONTROL) {
        mButton += 10 * pMods;
    }
    // reset UI transforms upon mouse middle click
    if (pButton == GLFW_MOUSE_BUTTON_MIDDLE && pMods == GLFW_MOD_CONTROL && pAction == GLFW_PRESS) {
        int r, c;
        getViewIds(&r, &c);
        glm::mat4& mvp = mMVPs[r+c*mRows];
        mvp = glm::mat4(1.0f);
    }
}

void Widget::pollEvents()
{
    glfwPollEvents();
}

void Widget::resizePixelBuffers()
{
    if (mFramePBO!=0)
        glDeleteBuffers(1, &mFramePBO);

    uint w = mWidth;
    uint h = mHeight;

    glGenBuffers(1, &mFramePBO);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, mFramePBO);
    glBufferData(GL_PIXEL_PACK_BUFFER, w*h*4*sizeof(internal::uchar), 0, GL_DYNAMIC_READ);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
}

}

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
#include <gl_native_handles.hpp>

#include <glm/gtc/matrix_transform.hpp>

using glm::rotate;
using glm::translate;
using glm::scale;

#include <algorithm>
#include <cmath>
#include <iostream>

#define GLFW_THROW_ERROR(msg, err) \
    FG_ERROR("Windows Constructor: "#msg, err)

namespace forge
{
namespace wtk
{

void initWindowToolkit()
{
    if (!glfwInit()) {
        std::cerr << "ERROR: GLFW wasn't able to initalize\n";
        GLFW_THROW_ERROR("GLFW initilization failed", FG_ERR_GL_ERROR);
    }
}

void destroyWindowToolkit()
{
    glfwTerminate();
}

const glm::mat4 Widget::findTransform(const MatrixHashMap& pMap, const float pX, const float pY)
{
    for (auto it: pMap) {
        const CellIndex& idx = it.first;
        const glm::mat4& mat  = it.second;

        const int rows = std::get<0>(idx);
        const int cols = std::get<1>(idx);

        const int cellWidth  = mWidth/cols;
        const int cellHeight = mHeight/rows;

        const int x = int(pX) / cellWidth;
        const int y = int(pY) / cellHeight;
        const int i = x + y * cols;
        if (i==std::get<2>(idx)) {
            return mat;
        }
    }

    return IDENTITY;
}

const glm::mat4 Widget::getCellViewMatrix(const float pXPos, const float pYPos)
{
    return findTransform(mViewMatrices, pXPos, pYPos);
}

const glm::mat4 Widget::getCellOrientationMatrix(const float pXPos, const float pYPos)
{
    return findTransform(mOrientMatrices, pXPos, pYPos);
}

void Widget::setTransform(MatrixHashMap& pMap, const float pX, const float pY, const glm::mat4 &pMat)
{
    for (auto it: pMap) {
        const CellIndex& idx = it.first;

        const int rows = std::get<0>(idx);
        const int cols = std::get<1>(idx);

        const int cellWidth  = mWidth/cols;
        const int cellHeight = mHeight/rows;

        const int x = int(pX) / cellWidth;
        const int y = int(pY) / cellHeight;
        const int i = x + y * cols;
        if (i==std::get<2>(idx)) {
            pMap[idx] = pMat;
        }
    }
}

void Widget::setCellViewMatrix(const float pXPos, const float pYPos, const glm::mat4& pMatrix)
{
    return setTransform(mViewMatrices, pXPos, pYPos, pMatrix);
}

void Widget::setCellOrientationMatrix(const float pXPos, const float pYPos, const glm::mat4& pMatrix)
{
    return setTransform(mOrientMatrices, pXPos, pYPos, pMatrix);
}


void Widget::resetViewMatrices()
{
    for (auto it: mViewMatrices)
        it.second = IDENTITY;
}


void Widget::resetOrientationMatrices()
{
    for (auto it: mOrientMatrices)
        it.second = IDENTITY;
}

Widget::Widget()
    : mWindow(NULL), mClose(false), mLastXPos(0), mLastYPos(0), mButton(-1),
    mWidth(512), mHeight(512)
{
}

Widget::Widget(int pWidth, int pHeight, const char* pTitle,
               const std::unique_ptr<Widget> &pWidget, const bool invisible)
    : mWindow(NULL), mClose(false), mLastXPos(0), mLastYPos(0), mButton(-1)
{
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
    mWindow = glfwCreateWindow(pWidth, pHeight,
                               (pTitle!=nullptr ? pTitle : "Forge-Demo"), nullptr,
                               (pWidget ? pWidget->getNativeHandle(): nullptr));

    if (!mWindow) {
        std::cerr<<"Error: Could not Create GLFW Window!\n";
        GLFW_THROW_ERROR("GLFW window creation failed", FG_ERR_GL_ERROR);
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
        static_cast<Widget*>(glfwGetWindowUserPointer(w))->cursorHandler(float(xpos), float(ypos));
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
    return opengl::getCurrentContextHandle();
}

long long Widget::getDisplayHandle()
{
    return opengl::getCurrentDisplayHandle();
}

GLADloadproc Widget::getProcAddr()
{
    return reinterpret_cast<GLADloadproc>(glfwGetProcAddress);
}

void Widget::setTitle(const char* pTitle)
{
    glfwSetWindowTitle(mWindow, (pTitle!=nullptr ? pTitle : "Forge-Demo"));
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

void Widget::resizeHandler(int pWidth, int pHeight)
{
    mWidth      = pWidth;
    mHeight     = pHeight;
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

    const glm::mat4 viewMat = getCellViewMatrix(pXPos, pYPos);

    if (mButton == GLFW_MOUSE_BUTTON_LEFT) {
        // Translate
        glm::mat4 vMat = translate(viewMat, glm::vec3(-deltaX, deltaY, 0.0f) * SPEED);

        setCellViewMatrix(pXPos, pYPos, vMat);

    } else if (mButton == GLFW_MOUSE_BUTTON_LEFT + 10 * GLFW_MOD_ALT ||
               mButton == GLFW_MOUSE_BUTTON_LEFT + 10 * GLFW_MOD_CONTROL) {
        // Zoom
        if(deltaY != 0.0f) {
            if(deltaY < 0.0f) {
                deltaY = 1.0f / (-deltaY);
            }
            glm::mat4 vMat = scale(viewMat, glm::vec3(pow(deltaY, SPEED)));

            setCellViewMatrix(pXPos, pYPos, vMat);
        }
    } else if (mButton == GLFW_MOUSE_BUTTON_RIGHT) {
        const glm::mat4 orientationMat = getCellOrientationMatrix(pXPos, pYPos);

        // Rotation
        int width, height;
        glfwGetWindowSize(mWindow, &width, &height);

        if (mLastXPos != pXPos || mLastYPos != pYPos) {
            glm::vec3 op1 = trackballPoint(mLastXPos, mLastYPos, float(width), float(height));
            glm::vec3 op2 = trackballPoint(pXPos, pYPos, float(width), float(height));

            float angle = std::acos(std::min(1.0f, glm::dot(op1, op2)));

            glm::vec3 axisInCamCoord = glm::cross(op1, op2);

            glm::mat3 camera2object = glm::inverse(glm::mat3(viewMat));
            glm::vec3 axisInObjCoord = camera2object * axisInCamCoord;

            glm::mat4 oMat = glm::rotate(orientationMat, glm::degrees(angle), axisInObjCoord);

            setCellOrientationMatrix(pXPos, pYPos, oMat);
        }
    }

    mLastXPos = pXPos;
    mLastYPos = pYPos;
}

void Widget::mouseButtonHandler(int pButton, int pAction, int pMods)
{
    double x, y;
    glfwGetCursorPos(mWindow, &x, &y);
    mLastXPos = float(x);
    mLastYPos = float(y);

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
        setCellViewMatrix(float(x), float(y), IDENTITY);
        setCellOrientationMatrix(float(x), float(y), IDENTITY);
    }
}

void Widget::pollEvents()
{
    glfwPollEvents();
}

const glm::mat4 Widget::getViewMatrix(const CellIndex& pIndex)
{
    if (mViewMatrices.find(pIndex)==mViewMatrices.end()) {
        mViewMatrices.emplace(pIndex, IDENTITY);
    }
    return mViewMatrices[pIndex];
}

const glm::mat4 Widget::getOrientationMatrix(const CellIndex& pIndex)
{
    if (mOrientMatrices.find(pIndex)==mOrientMatrices.end()) {
        mOrientMatrices.emplace(pIndex, IDENTITY);
    }
    return mOrientMatrices[pIndex];
}

}
}

/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/defines.hpp>
#include <common/err_handling.hpp>
#include <gl_native_handles.hpp>
#include <glfw/window.hpp>

#include <glm/gtc/epsilon.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <cmath>
#include <utility>

using glm::epsilonNotEqual;
using glm::make_vec4;
using glm::mat4;
using glm::rotate;
using glm::scale;
using glm::translate;
using glm::vec2;
using glm::vec3;
using glm::vec4;
using std::floor;
using std::get;
using std::make_tuple;
using std::tuple;

using namespace forge::common;

#define GLFW_THROW_ERROR(msg, err) FG_ERROR("Windows Constructor: " #msg, err)

namespace forge {
namespace wtk {

void initWindowToolkit() {
    if (!glfwInit()) {
        std::cerr << "ERROR: GLFW wasn't able to initalize\n";
        GLFW_THROW_ERROR("GLFW initilization failed", FG_ERR_GL_ERROR);
    }
}

void destroyWindowToolkit() { glfwTerminate(); }

tuple<vec3, vec2> getCellCoordsDims(const vec2& pos, const CellIndex& idx,
                                    const vec2& dims) {
    const int rows = get<0>(idx);
    const int cols = get<1>(idx);
    const int cw   = dims[0] / cols;
    const int ch   = dims[1] / rows;
    const int x    = static_cast<int>(floor(pos[0] / static_cast<double>(cw)));
    const int y    = static_cast<int>(floor(pos[1] / static_cast<double>(ch)));
    return make_tuple(vec3(x * cw, y * ch, x + y * cols), vec2(cw, ch));
}

const vec4 Widget::getCellViewport(const vec2& pos) {
    // Either of the transformation matrix maps are fine for figuring
    // out the viewport corresponding to the current mouse position
    // Here I am using mOrientMatrices map
    vec4 retVal(0, 0, mWidth, mHeight);
    for (auto& it : mOrientMatrices) {
        const CellIndex& idx = it.first;
        auto coordsAndDims = getCellCoordsDims(pos, idx, vec2(mWidth, mHeight));
        if (get<0>(coordsAndDims)[2] == std::get<2>(idx)) {
            retVal = vec4(get<0>(coordsAndDims)[0], get<0>(coordsAndDims)[1],
                          get<1>(coordsAndDims));
            break;
        }
    }
    return retVal;
}

const mat4 Widget::findTransform(const MatrixHashMap& pMap, const double pX,
                                 const double pY) {
    for (auto it : pMap) {
        const CellIndex& idx = it.first;
        const mat4& mat      = it.second;
        auto coordsAndDims =
            getCellCoordsDims(vec2(pX, pY), idx, vec2(mWidth, mHeight));
        if (get<0>(coordsAndDims)[2] == std::get<2>(idx)) { return mat; }
    }
    return IDENTITY;
}

const mat4 Widget::getCellViewMatrix(const double pXPos, const double pYPos) {
    return findTransform(mViewMatrices, pXPos, pYPos);
}

const mat4 Widget::getCellOrientationMatrix(const double pXPos,
                                            const double pYPos) {
    return findTransform(mOrientMatrices, pXPos, pYPos);
}

void Widget::setTransform(MatrixHashMap& pMap, const double pX, const double pY,
                          const mat4& pMat) {
    for (auto it : pMap) {
        const CellIndex& idx = it.first;
        auto coordsAndDims =
            getCellCoordsDims(vec2(pX, pY), idx, vec2(mWidth, mHeight));
        if (get<0>(coordsAndDims)[2] == std::get<2>(idx)) {
            pMap[idx] = pMat;
            return;
        }
    }
}

void Widget::setCellViewMatrix(const double pXPos, const double pYPos,
                               const mat4& pMatrix) {
    return setTransform(mViewMatrices, pXPos, pYPos, pMatrix);
}

void Widget::setCellOrientationMatrix(const double pXPos, const double pYPos,
                                      const mat4& pMatrix) {
    return setTransform(mOrientMatrices, pXPos, pYPos, pMatrix);
}

void Widget::resetViewMatrices() {
    for (auto it : mViewMatrices) it.second = IDENTITY;
}

void Widget::resetOrientationMatrices() {
    for (auto it : mOrientMatrices) it.second = IDENTITY;
}

Widget::Widget()
    : mWindow(NULL)
    , mClose(false)
    , mLastPos(0, 0)
    , mButton(-1)
    , mRotationFlag(false)
    , mWidth(512)
    , mHeight(512) {}

Widget::Widget(int pWidth, int pHeight, const char* pTitle,
               const std::unique_ptr<Widget>& pWidget, const bool invisible)
    : mWindow(NULL)
    , mClose(false)
    , mLastPos(0, 0)
    , mButton(-1)
    , mRotationFlag(false) {
    auto wndErrCallback = [](int errCode, const char* pDescription) {
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
    mWindow = glfwCreateWindow(
        pWidth, pHeight, (pTitle != nullptr ? pTitle : "Forge-Demo"), nullptr,
        (pWidget ? pWidget->getNativeHandle() : nullptr));

    if (!mWindow) {
        std::cerr << "Error: Could not Create GLFW Window!\n";
        GLFW_THROW_ERROR("GLFW window creation failed", FG_ERR_GL_ERROR);
    }

    glfwSetWindowUserPointer(mWindow, this);

    auto rsCallback = [](GLFWwindow* w, int pWidth, int pHeight) {
        static_cast<Widget*>(glfwGetWindowUserPointer(w))
            ->resizeHandler(pWidth, pHeight);
    };

    auto kbCallback = [](GLFWwindow* w, int pKey, int pScancode, int pAction,
                         int pMods) {
        static_cast<Widget*>(glfwGetWindowUserPointer(w))
            ->keyboardHandler(pKey, pScancode, pAction, pMods);
    };

    auto closeCallback = [](GLFWwindow* w) {
        static_cast<Widget*>(glfwGetWindowUserPointer(w))->hide();
    };

    auto cursorCallback = [](GLFWwindow* w, double xpos, double ypos) {
        static_cast<Widget*>(glfwGetWindowUserPointer(w))
            ->cursorHandler(xpos, ypos);
    };

    auto mouseButtonCallback = [](GLFWwindow* w, int button, int action,
                                  int mods) {
        static_cast<Widget*>(glfwGetWindowUserPointer(w))
            ->mouseButtonHandler(button, action, mods);
    };

    glfwSetFramebufferSizeCallback(mWindow, rsCallback);
    glfwSetWindowCloseCallback(mWindow, closeCallback);
    glfwSetKeyCallback(mWindow, kbCallback);
    glfwSetCursorPosCallback(mWindow, cursorCallback);
    glfwSetMouseButtonCallback(mWindow, mouseButtonCallback);

    glfwGetFramebufferSize(mWindow, &mWidth, &mHeight);

    // Set Hand cursor for Rotation and Zoom Modes
    mRotationCursor = glfwCreateStandardCursor(GLFW_HAND_CURSOR);
    mZoomCursor     = glfwCreateStandardCursor(GLFW_VRESIZE_CURSOR);
}

Widget::~Widget() {
    if (mWindow) glfwDestroyWindow(mWindow);
    if (mRotationCursor) glfwDestroyCursor(mRotationCursor);
    if (mZoomCursor) glfwDestroyCursor(mZoomCursor);
}

GLFWwindow* Widget::getNativeHandle() const { return mWindow; }

void Widget::makeContextCurrent() const { glfwMakeContextCurrent(mWindow); }

long long Widget::getGLContextHandle() {
    return opengl::getCurrentContextHandle();
}

long long Widget::getDisplayHandle() {
    return opengl::getCurrentDisplayHandle();
}

GLADloadproc Widget::getProcAddr() {
    return reinterpret_cast<GLADloadproc>(glfwGetProcAddress);
}

void Widget::setTitle(const char* pTitle) {
    glfwSetWindowTitle(mWindow, (pTitle != nullptr ? pTitle : "Forge-Demo"));
}

void Widget::setPos(int pX, int pY) { glfwSetWindowPos(mWindow, pX, pY); }

void Widget::setSize(unsigned pW, unsigned pH) {
    glfwSetWindowSize(mWindow, pW, pH);
}

void Widget::swapBuffers() { glfwSwapBuffers(mWindow); }

void Widget::hide() {
    mClose = true;
    glfwHideWindow(mWindow);
}

void Widget::show() {
    mClose = false;
    glfwShowWindow(mWindow);
}

bool Widget::close() { return mClose; }

void Widget::resetCloseFlag() {
    if (mClose == true) { show(); }
}

void Widget::resizeHandler(int pWidth, int pHeight) {
    mWidth  = pWidth;
    mHeight = pHeight;
}

void Widget::keyboardHandler(int pKey, int pScancode, int pAction, int pMods) {
    if (pKey == GLFW_KEY_ESCAPE && pAction == GLFW_PRESS) { hide(); }
}

void Widget::cursorHandler(const double pXPos, const double pYPos) {
    constexpr auto ZOOM_ENABLER =
        GLFW_MOUSE_BUTTON_LEFT + 10 * GLFW_MOD_CONTROL;

    const vec2 currPos(pXPos, pYPos);

    if (mButtonAction == GLFW_PRESS) {
        auto delta   = mLastPos - currPos;
        auto viewMat = getCellViewMatrix(pXPos, pYPos);

        if (mButton == GLFW_MOUSE_BUTTON_LEFT) {
            // Translate
            mat4 vMat = translate(viewMat,
                                  vec3(-delta[0], delta[1], 0.0f) * MOVE_SPEED);

            setCellViewMatrix(pXPos, pYPos, vMat);
        } else if (mButton == ZOOM_ENABLER) {
            // Zoom
            double dy = delta[1];
            if (!(std::abs(dy) < EPSILON)) {
                if (dy < 0.0f) { dy = -1.0f / dy; }
                mat4 vMat = scale(viewMat, vec3(pow(dy, ZOOM_SPEED)));
                setCellViewMatrix(pXPos, pYPos, vMat);
            }
        } else if (mButton == GLFW_MOUSE_BUTTON_RIGHT) {
            // Rotation
            auto compCmp = epsilonNotEqual(mLastPos, currPos, vec2(EPSILON));
            if (compCmp[0] || compCmp[1]) {
                const mat4 oMat = getCellOrientationMatrix(pXPos, pYPos);
                const vec4 vprt = getCellViewport(currPos);
                auto rotParams =
                    calcRotationFromArcBall(mLastPos, currPos, vprt);

                setCellOrientationMatrix(
                    pXPos, pYPos,
                    rotate(oMat, rotParams.second, rotParams.first));
            }
        }
    }
    mLastPos = currPos;
}

void Widget::mouseButtonHandler(int pButton, int pAction, int pMods) {
    mButton       = pButton;
    mButtonAction = pAction;

    const bool isZoomModifierOn = pMods == GLFW_MOD_CONTROL;

    double x, y;
    glfwGetCursorPos(mWindow, &x, &y);
    auto pos = vec2(x, y);

    if (mButtonAction == GLFW_PRESS) {
        if (mButton == GLFW_MOUSE_BUTTON_RIGHT) {
            glfwSetCursor(mWindow, mRotationCursor);
            mRotationFlag = true;
        } else if (mButton == GLFW_MOUSE_BUTTON_LEFT && isZoomModifierOn) {
            glfwSetCursor(mWindow, mZoomCursor);
        }
        mLastPos = pos;
    } else if (mButtonAction == GLFW_RELEASE) {
        mRotationFlag = false;
        glfwSetCursor(mWindow, NULL);
    }
    mButton += (10 * pMods * isZoomModifierOn);

    // reset UI transforms upon mouse middle click
    if (pMods == GLFW_MOD_CONTROL && pButton == GLFW_MOUSE_BUTTON_MIDDLE &&
        pAction == GLFW_PRESS) {
        setCellViewMatrix(pos[0], pos[1], IDENTITY);
        setCellOrientationMatrix(pos[0], pos[1], IDENTITY);
        mButton       = -1;
        mButtonAction = -1;
    }
}

void Widget::pollEvents() { glfwPollEvents(); }

const mat4 Widget::getViewMatrix(const CellIndex& pIndex) {
    if (mViewMatrices.find(pIndex) == mViewMatrices.end()) {
        mViewMatrices.emplace(pIndex, IDENTITY);
    }
    return mViewMatrices[pIndex];
}

const mat4 Widget::getOrientationMatrix(const CellIndex& pIndex) {
    if (mOrientMatrices.find(pIndex) == mOrientMatrices.end()) {
        mOrientMatrices.emplace(pIndex, IDENTITY);
    }
    return mOrientMatrices[pIndex];
}

glm::vec2 Widget::getCursorPos() const {
    double xp, yp;
    glfwGetCursorPos(mWindow, &xp, &yp);
    return {xp, yp};
}

}  // namespace wtk
}  // namespace forge

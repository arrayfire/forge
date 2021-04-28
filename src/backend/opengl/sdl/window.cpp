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
#include <sdl/window.hpp>

#include <glm/gtc/epsilon.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <cmath>

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

#define SDL_THROW_ERROR(msg, err) FG_ERROR("Window constructor " #msg, err)

namespace forge {
namespace wtk {

void initWindowToolkit() {
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "ERROR: SDL wasn't able to initalize\n";
        SDL_THROW_ERROR("SDL initilization failed", FG_ERR_GL_ERROR);
    }
}

void destroyWindowToolkit() { SDL_Quit(); }

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
    : mWindow(nullptr)
    , mDefaultCursor(nullptr)
    , mRotationCursor(nullptr)
    , mZoomCursor(nullptr)
    , mMoveCursor(nullptr)
    , mClose(false)
    , mLastPos(0, 0)
    , mRotationFlag(false)
    , mWidth(512)
    , mHeight(512) {}

Widget::Widget(int pWidth, int pHeight, const char* pTitle,
               const std::unique_ptr<Widget>& pWidget, const bool invisible)
    : mWindow(nullptr)
    , mDefaultCursor(nullptr)
    , mRotationCursor(nullptr)
    , mZoomCursor(nullptr)
    , mMoveCursor(nullptr)
    , mClose(false)
    , mLastPos(0, 0)
    , mRotationFlag(false) {
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK,
                        SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

    if (pWidget) {
        pWidget->makeContextCurrent();
        SDL_GL_SetAttribute(SDL_GL_SHARE_WITH_CURRENT_CONTEXT, 1);
    } else {
        // SDL_GL_MakeCurrent(NULL, NULL);
        SDL_GL_SetAttribute(SDL_GL_SHARE_WITH_CURRENT_CONTEXT, 0);
    }

    mWindow = SDL_CreateWindow(
        (pTitle != nullptr ? pTitle : "Forge-Demo"), SDL_WINDOWPOS_UNDEFINED,
        SDL_WINDOWPOS_UNDEFINED, pWidth, pHeight,
        (invisible ? SDL_WINDOW_OPENGL | SDL_WINDOW_HIDDEN
                   : SDL_WINDOW_OPENGL) |
            SDL_WINDOW_RESIZABLE);
    if (mWindow == NULL) {
        std::cerr << "Error: Could not Create SDL Window!" << SDL_GetError()
                  << std::endl;
        SDL_THROW_ERROR("SDL window creation failed", FG_ERR_GL_ERROR);
    }

    mContext = SDL_GL_CreateContext(mWindow);
    if (mContext == NULL) {
        std::cerr << "Error: Could not OpenGL context!" << SDL_GetError()
                  << std::endl;
        SDL_THROW_ERROR("OpenGL context creation failed", FG_ERR_GL_ERROR);
    }

    SDL_GL_SetSwapInterval(1);
    mWindowId = SDL_GetWindowID(mWindow);
    SDL_GetWindowSize(mWindow, &mWidth, &mHeight);

    // Set Hand cursor for Rotation and Zoom Modes
    mDefaultCursor  = SDL_GetDefaultCursor();
    mRotationCursor = SDL_CreateSystemCursor(SDL_SYSTEM_CURSOR_HAND);
    mZoomCursor     = SDL_CreateSystemCursor(SDL_SYSTEM_CURSOR_SIZENS);
    mMoveCursor     = SDL_CreateSystemCursor(SDL_SYSTEM_CURSOR_SIZEALL);
}

Widget::~Widget() {
    if (mContext) SDL_GL_DeleteContext(mContext);
    if (mWindow) SDL_DestroyWindow(mWindow);
    if (mRotationCursor) SDL_FreeCursor(mRotationCursor);
    if (mZoomCursor) SDL_FreeCursor(mZoomCursor);
}

SDL_Window* Widget::getNativeHandle() const { return mWindow; }

void Widget::makeContextCurrent() const {
    SDL_GL_MakeCurrent(mWindow, mContext);
}

long long Widget::getGLContextHandle() {
    return opengl::getCurrentContextHandle();
}

long long Widget::getDisplayHandle() {
    return opengl::getCurrentDisplayHandle();
}

GLADloadproc Widget::getProcAddr() {
    return static_cast<GLADloadproc>(SDL_GL_GetProcAddress);
}

void Widget::setTitle(const char* pTitle) {
    SDL_SetWindowTitle(mWindow, (pTitle != nullptr ? pTitle : "Forge-Demo"));
}

void Widget::setPos(int pX, int pY) { SDL_SetWindowPosition(mWindow, pX, pY); }

void Widget::setSize(unsigned pW, unsigned pH) {
    SDL_SetWindowSize(mWindow, pW, pH);
}

void Widget::swapBuffers() { SDL_GL_SwapWindow(mWindow); }

void Widget::hide() {
    mClose = true;
    SDL_HideWindow(mWindow);
}

void Widget::show() {
    mClose = false;
    SDL_ShowWindow(mWindow);
}

bool Widget::close() { return mClose; }

void Widget::resetCloseFlag() {
    if (mClose == true) { show(); }
}

void Widget::pollEvents() {
    SDL_Event evnt;
    while (SDL_PollEvent(&evnt)) {
        /* handle window events that are triggered
           when the window with window id 'mWindowId' is in focus
           */
        if (evnt.key.windowID == mWindowId) {
            // Window Events
            if (evnt.type == SDL_WINDOWEVENT) {
                switch (evnt.window.event) {
                    case SDL_WINDOWEVENT_CLOSE: mClose = true; break;
                    case SDL_WINDOWEVENT_RESIZED:
                        mWidth  = evnt.window.data1;
                        mHeight = evnt.window.data2;
                        break;
                }
            }
            // Keyboard Events
            if (evnt.type == SDL_KEYDOWN) {
                switch (evnt.key.keysym.sym) {
                    case SDLK_ESCAPE: mClose = true; break;
                    default: break;
                }
            }
            // Mouse Events
            auto kbState = SDL_GetKeyboardState(NULL);
            bool isCtrl =
                kbState[SDL_SCANCODE_LCTRL] || kbState[SDL_SCANCODE_RCTRL];

            if (evnt.type == SDL_MOUSEMOTION) {
                auto currPos = vec2(evnt.motion.x, evnt.motion.y);
                if (evnt.motion.state == SDL_BUTTON_LMASK) {
                    auto viewMat = getCellViewMatrix(currPos[0], currPos[1]);
                    auto delta   = mLastPos - currPos;
                    if (isCtrl) {
                        // Zoom
                        double dy = delta[1];
                        if (!(std::abs(dy) < EPSILON)) {
                            if (dy < 0.0f) { dy = -1.0 / dy; }
                            mat4 vMat =
                                scale(viewMat, vec3(pow(dy, ZOOM_SPEED)));
                            setCellViewMatrix(currPos[0], currPos[1], vMat);
                        }
                    } else {
                        // Translate
                        mat4 vMat =
                            translate(viewMat, vec3(-delta[0], delta[1], 0.0f) *
                                                   MOVE_SPEED);
                        setCellViewMatrix(currPos[0], currPos[1], vMat);
                    }
                } else if (evnt.motion.state == SDL_BUTTON_RMASK) {
                    // Rotations
                    auto compCmp =
                        epsilonNotEqual(mLastPos, currPos, vec2(EPSILON));
                    if (compCmp[0] || compCmp[1]) {
                        const mat4 oMat =
                            getCellOrientationMatrix(currPos[0], currPos[1]);
                        const vec4 vprt = getCellViewport(currPos);
                        auto rotParams =
                            calcRotationFromArcBall(mLastPos, currPos, vprt);

                        setCellOrientationMatrix(
                            currPos[0], currPos[1],
                            rotate(oMat, rotParams.second, rotParams.first));
                    }
                }
                mLastPos = currPos;
            } else if (evnt.type == SDL_MOUSEBUTTONDOWN) {
                auto button = evnt.button.button;
                if (button == SDL_BUTTON_LEFT && isCtrl) {
                    // Zoom left mouse button special case first
                    SDL_SetCursor(mZoomCursor);
                } else if (button == SDL_BUTTON_LEFT) {
                    // Translation
                    SDL_SetCursor(mMoveCursor);
                } else if (button == SDL_BUTTON_RIGHT) {
                    // Rotation
                    mRotationFlag = true;
                    SDL_SetCursor(mRotationCursor);
                } else if (button == SDL_BUTTON_MIDDLE && isCtrl) {
                    // reset UI transforms upon mouse middle click
                    setCellViewMatrix(evnt.button.x, evnt.button.y, IDENTITY);
                    setCellOrientationMatrix(evnt.button.x, evnt.button.y,
                                             IDENTITY);
                    mRotationFlag = false;
                    SDL_SetCursor(mDefaultCursor);
                }
                mLastPos = vec2(evnt.button.x, evnt.button.y);
            } else if (evnt.type == SDL_MOUSEBUTTONUP) {
                mRotationFlag = false;
                SDL_SetCursor(mDefaultCursor);
            }
        }
    }
}

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
    int xp, yp;
    SDL_GetMouseState(&xp, &yp);
    return {xp, yp};
}

}  // namespace wtk
}  // namespace forge

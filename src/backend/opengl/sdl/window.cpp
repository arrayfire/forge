/*******************************************************
* Copyright (c) 2015-2019, ArrayFire
* All rights reserved.
*
* This file is distributed under 3-clause BSD license.
* The complete license agreement can be obtained at:
* http://arrayfire.com/licenses/BSD-3-Clause
********************************************************/

#include <common.hpp>
#include <sdl/window.hpp>
#include <gl_native_handles.hpp>

#include <glm/gtc/matrix_transform.hpp>
#include <iostream>

using namespace gl;

using glm::rotate;
using glm::translate;
using glm::scale;

#define SDL_THROW_ERROR(msg, err) \
    FG_ERROR("Window constructor "#msg,err)

namespace forge
{
namespace wtk
{

void initWindowToolkit()
{
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "ERROR: SDL wasn't able to initalize\n";
        SDL_THROW_ERROR("SDL initilization failed", FG_ERR_GL_ERROR);
    }
}

void destroyWindowToolkit()
{
    SDL_Quit();
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
    : mWindow(nullptr), mClose(false), mLastXPos(0), mLastYPos(0), mButton(-1),
    mWidth(512), mHeight(512), mFramePBO(0)
{
}

Widget::Widget(int pWidth, int pHeight, const char* pTitle, const Widget* pWindow, const bool invisible)
    : mWindow(nullptr), mClose(false), mLastXPos(0), mLastYPos(0), mButton(-1), mFramePBO(0)
{
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

    if (pWindow != nullptr) {
        pWindow->makeContextCurrent();
        SDL_GL_SetAttribute(SDL_GL_SHARE_WITH_CURRENT_CONTEXT, 1);
    } else {
        //SDL_GL_MakeCurrent(NULL, NULL);
        SDL_GL_SetAttribute(SDL_GL_SHARE_WITH_CURRENT_CONTEXT, 0);
    }

    mWindow = SDL_CreateWindow(
                            (pTitle!=nullptr ? pTitle : "Forge-Demo"),
                            SDL_WINDOWPOS_UNDEFINED,
                            SDL_WINDOWPOS_UNDEFINED,
                            pWidth, pHeight,
                            (invisible ? SDL_WINDOW_OPENGL | SDL_WINDOW_HIDDEN
                             : SDL_WINDOW_OPENGL) | SDL_WINDOW_RESIZABLE
                            );
    if (mWindow==NULL) {
        std::cerr<<"Error: Could not Create SDL Window!"<< SDL_GetError() << std::endl;
        SDL_THROW_ERROR("SDL window creation failed", FG_ERR_GL_ERROR);
    }

    mContext = SDL_GL_CreateContext(mWindow);
    if (mContext==NULL) {
        std::cerr<<"Error: Could not OpenGL context!" << SDL_GetError() << std::endl;
        SDL_THROW_ERROR("OpenGL context creation failed", FG_ERR_GL_ERROR);
    }

    SDL_GL_SetSwapInterval(1);
    mWindowId = SDL_GetWindowID(mWindow);
    SDL_GetWindowSize(mWindow, &mWidth, &mHeight);
}

Widget::~Widget()
{
    glDeleteBuffers(1, &mFramePBO);

    SDL_DestroyWindow(mWindow);
    SDL_GL_DeleteContext(mContext);
}

SDL_Window* Widget::getNativeHandle() const
{
    return mWindow;
}

void Widget::makeContextCurrent() const
{
    SDL_GL_MakeCurrent(mWindow, mContext);
}

long long Widget::getGLContextHandle()
{
    return opengl::getCurrentContextHandle();
}

long long Widget::getDisplayHandle()
{
    return opengl::getCurrentDisplayHandle();
}

void Widget::setTitle(const char* pTitle)
{
    SDL_SetWindowTitle(mWindow, (pTitle!=nullptr ? pTitle : "Forge-Demo"));
}

void Widget::setPos(int pX, int pY)
{
    SDL_SetWindowPosition(mWindow, pX, pY);
}

void Widget::setSize(unsigned pW, unsigned pH)
{
    SDL_SetWindowSize(mWindow, pW, pH);
}

void Widget::swapBuffers()
{
    SDL_GL_SwapWindow(mWindow);

    glReadBuffer(GL_FRONT);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, mFramePBO);
    glReadPixels(0, 0, mWidth, mHeight, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
}

void Widget::hide()
{
    mClose = true;
    SDL_HideWindow(mWindow);
}

void Widget::show()
{
    mClose = false;
    SDL_ShowWindow(mWindow);
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

void Widget::pollEvents()
{
    static const float SPEED = 0.005f;
    SDL_Event evnt;

    while (SDL_PollEvent(&evnt)) {
        /* handle window events that are triggered
           when the window with window id 'mWindowId' is in focus
           */
        if (evnt.key.windowID == mWindowId) {
            if (evnt.type == SDL_WINDOWEVENT) {
                switch(evnt.window.event) {
                    case SDL_WINDOWEVENT_CLOSE:
                        mClose = true;
                        break;
                    case SDL_WINDOWEVENT_RESIZED:
                        mWidth      = evnt.window.data1;
                        mHeight     = evnt.window.data2;
                        resizePixelBuffers();
                        break;
                }
            }

            if (evnt.type == SDL_KEYDOWN) {
                switch(evnt.key.keysym.sym) {
                    case SDLK_ESCAPE:
                        mClose = true; break;
                    default:
                        mMod = evnt.key.keysym.sym;
                        break;
                }
            } else if (evnt.type == SDL_KEYUP) {
                mMod = -1;
            }

            int x, y;
            SDL_GetMouseState(&x, &y);
            // reset UI transforms upon mouse middle click
            if(evnt.type == SDL_MOUSEBUTTONUP) {
                if(evnt.button.button == SDL_BUTTON_MIDDLE &&
                        (mMod == SDLK_LCTRL || mMod==SDLK_RCTRL)) {
                    setCellViewMatrix(x, y, IDENTITY);
                    setCellOrientationMatrix(x, y, IDENTITY);
                }
            }

            const glm::mat4 viewMat = getCellViewMatrix(x, y);

            if(evnt.type == SDL_MOUSEMOTION) {
                if(evnt.motion.state == SDL_BUTTON_LMASK) {
                    double deltaX = -evnt.motion.xrel;
                    double deltaY = -evnt.motion.yrel;

                    glm::mat4 vMat(1);

                    if (mMod==SDLK_LALT || mMod==SDLK_RALT || mMod==SDLK_LCTRL || mMod==SDLK_RCTRL) {
                        // Zoom
                        if(deltaY != 0) {
                            if(deltaY < 0) {
                                deltaY = 1.0 / (-deltaY);
                            }
                            vMat = scale(viewMat, glm::vec3(pow(deltaY, SPEED)));
                        }
                    } else {
                        // Translate
                        vMat = translate(viewMat, glm::vec3(-deltaX, deltaY, 0.0f) * SPEED);
                    }
                    setCellViewMatrix(x, y, vMat);
                } else if (evnt.motion.state == SDL_BUTTON_RMASK) {
                    const glm::mat4 orientationMat = getCellOrientationMatrix(x, y);
                    // Rotations
                    int width, height;
                    SDL_GetWindowSize(mWindow, &width, &height);

                    int xPos = evnt.motion.x;
                    int yPos = evnt.motion.y;

                    if (mLastXPos != xPos || mLastYPos != yPos) {
                        glm::vec3 op1 = trackballPoint(mLastXPos, mLastYPos, width, height);
                        glm::vec3 op2 = trackballPoint(xPos, yPos, width, height);

                        float angle = std::acos(std::min(1.0f, glm::dot(op1, op2)));

                        glm::vec3 axisInCamCoord = glm::cross(op1, op2);

                        glm::mat3 camera2object = glm::inverse(glm::mat3(viewMat));
                        glm::vec3 axisInObjCoord = camera2object * axisInCamCoord;

                        glm::mat4 oMat = glm::rotate(orientationMat,
                                                     glm::degrees(angle), axisInObjCoord);
                        setCellOrientationMatrix(x, y, oMat);
                    }
                }

                mLastXPos = evnt.motion.x;
                mLastYPos = evnt.motion.y;
            }
        }
    }
}

void Widget::resizePixelBuffers()
{
    if (mFramePBO!=0)
        glDeleteBuffers(1, &mFramePBO);

    uint w = mWidth;
    uint h = mHeight;

    glGenBuffers(1, &mFramePBO);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, mFramePBO);
    glBufferData(GL_PIXEL_PACK_BUFFER, w*h*4*sizeof(uchar), 0, GL_DYNAMIC_READ);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
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

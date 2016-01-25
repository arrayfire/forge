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

#include <glm/gtc/matrix_transform.hpp>

#ifndef OS_WIN
#include <GL/glx.h>
#else
#include <windows.h>
#endif
#include <iostream>

using glm::rotate;
using glm::translate;
using glm::scale;

#define SDL_THROW_ERROR(msg, err) \
    throw fg::Error("Window constructor", __LINE__, msg, err);

namespace wtk
{

Widget::Widget()
    : mWindow(nullptr), mClose(false)
{
}

Widget::Widget(int pWidth, int pHeight, const char* pTitle, const Widget* pWindow, const bool invisible)
    : mWindow(nullptr), mClose(false),
      mLastXPos(0), mLastYPos(0), mMVP(glm::mat4(1.0f)), mButton(-1), mMod(-1)
{
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "ERROR: SDL wasn't able to initalize\n";
        SDL_THROW_ERROR("SDL initilization failed", fg::FG_ERR_GL_ERROR)
    }

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
                            pTitle,
                            SDL_WINDOWPOS_UNDEFINED,
                            SDL_WINDOWPOS_UNDEFINED,
                            pWidth, pHeight,
                            (invisible ? SDL_WINDOW_OPENGL | SDL_WINDOW_HIDDEN
                             : SDL_WINDOW_OPENGL) | SDL_WINDOW_RESIZABLE
                            );
    if (mWindow==NULL) {
        std::cerr<<"Error: Could not Create SDL Window!"<< SDL_GetError() << std::endl;
        SDL_THROW_ERROR("sdl window creation failed", fg::FG_ERR_GL_ERROR)
    }

    mContext = SDL_GL_CreateContext(mWindow);
    if (mContext==NULL) {
        std::cerr<<"Error: Could not OpenGL context!" << SDL_GetError() << std::endl;
        SDL_THROW_ERROR("opengl context creation failed", fg::FG_ERR_GL_ERROR)
    }

    SDL_GL_SetSwapInterval(1);
    mWindowId = SDL_GetWindowID(mWindow);
}

Widget::~Widget()
{
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
#ifdef OS_WIN
    return reinterpret_cast<long long>(wglGetCurrentContext());
#endif
#ifdef OS_LNX
    return reinterpret_cast<long long>(glXGetCurrentContext());
#endif
}

long long Widget::getDisplayHandle()
{
#ifdef OS_WIN
    return reinterpret_cast<long long>(wglGetCurrentDC());
#endif
#ifdef OS_LNX
    return reinterpret_cast<long long>(glXGetCurrentDisplay());
#endif
}

void Widget::getFrameBufferSize(int* pW, int* pH)
{
    /* FIXME this needs to be framebuffer size */
    SDL_GetWindowSize(mWindow, pW, pH);
}

void Widget::setTitle(const char* pTitle)
{
    SDL_SetWindowTitle(mWindow, pTitle);
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
    SDL_PollEvent(&evnt);

    /* handle window events that are triggered
       when the window with window id 'mWindowId' is in focus
     */
    if (evnt.key.windowID == mWindowId) {
        if (evnt.type == SDL_WINDOWEVENT) {
            switch(evnt.window.event) {
                case SDL_WINDOWEVENT_CLOSE: mClose = true; break;
            }
        }

        if (evnt.type == SDL_KEYDOWN) {
            switch(evnt.key.keysym.sym) {
                case SDLK_ESCAPE: mClose = true     ; break;
                case SDLK_LALT  : mMod   = SDLK_LALT; break;
                case SDLK_RALT  : mMod   = SDLK_RALT; break;
            }
        } else if (evnt.type == SDL_KEYUP) {
            switch(evnt.key.keysym.sym) {
                case SDLK_LALT: mMod = -1; break;
                case SDLK_RALT: mMod = -1; break;
            }
        }

        if(evnt.type == SDL_MOUSEBUTTONUP) {
            if(evnt.button.button == SDL_BUTTON_MIDDLE && mMod == SDLK_LALT) {
                mMVP = glm::mat4(1.0f);
            }
        }

        if(evnt.type == SDL_MOUSEMOTION) {
            double deltaX = -evnt.motion.xrel;
            double deltaY = -evnt.motion.yrel;
            bool majorMoveDir = abs(deltaX) > abs(deltaY);  // True for Left-Right, False for Up-Down

            if(evnt.motion.state == SDL_BUTTON_LMASK &&
                   (mMod == SDLK_LALT || mMod == SDLK_RALT)) {
                // Zoom
                if(deltaY != 0) {
                    if(deltaY < 0) {
                        deltaY = 1.0 / (-deltaY);
                    }
                    mMVP = scale(mMVP, glm::vec3(pow(deltaY, SPEED)));
                }
            } else if (evnt.motion.state == SDL_BUTTON_LMASK) {
                // Translate
                mMVP = translate(mMVP, glm::vec3(-deltaX, deltaY, 0.0f) * SPEED);
            } else if (evnt.motion.state == SDL_BUTTON_RMASK) {
                // Rotations
                if (majorMoveDir) {
                    // Rotate about Y axis (left <-> right)
                    mMVP = rotate(mMVP, (float)(SPEED * deltaX), glm::vec3(0.0, 1.0, 0.0));
                } else {
                    // Rotate about X axis (up <-> down)glm::
                    mMVP = rotate(mMVP, (float)(SPEED * deltaY), glm::vec3(1.0, 0.0, 0.0));
                }
            }

            mLastXPos = evnt.motion.x;
            mLastYPos = evnt.motion.y;
        }
    }
}

}

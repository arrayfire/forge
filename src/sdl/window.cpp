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

#ifndef OS_WIN
#include <GL/glx.h>
#else
#include <windows.h>
#endif

#include <iostream>

#define SDL_THROW_ERROR(msg, err) \
    throw fg::Error("Window constructor", __LINE__, msg, err);

namespace wtk
{

Widget::Widget()
    : mWindow(nullptr), mClose(false)
{
}

Widget::Widget(int pWidth, int pHeight, const char* pTitle, const Widget* pWindow, const bool invisible)
    : mWindow(nullptr), mClose(false)
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
    SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 4);
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
                             : SDL_WINDOW_OPENGL)
                            );
    mContext = SDL_GL_CreateContext(mWindow);

    if (mWindow==NULL) {
        std::cerr<<"Error: Could not Create SDL Window!\n";
        SDL_THROW_ERROR("sdl window creation failed", fg::FG_ERR_GL_ERROR)
    }

    SDL_GL_SetSwapInterval(1);
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

void Widget::swapBuffers()
{
    SDL_GL_SwapWindow(mWindow);
}

void Widget::hide()
{
    SDL_HideWindow(mWindow);
}

void Widget::show()
{
    SDL_ShowWindow(mWindow);
}

bool Widget::close()
{
    return mClose;
}

void Widget::pollEvents()
{
    SDL_Event evnt;
    SDL_PollEvent(&evnt);

    if (evnt.type == SDL_WINDOWEVENT) {
        switch(evnt.window.event) {
            case SDL_WINDOWEVENT_CLOSE:
                mClose = true;
                break;
        }
    } else if (evnt.type == SDL_KEYDOWN) {
        switch(evnt.key.keysym.sym) {
            case SDLK_ESCAPE:
                mClose = true;
                break;
        }
    }
}

}

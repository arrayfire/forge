/*******************************************************
* Copyright (c) 2015-2019, ArrayFire
* All rights reserved.
*
* This file is distributed under 3-clause BSD license.
* The complete license agreement can be obtained at:
* http://arrayfire.com/licenses/BSD-3-Clause
********************************************************/

#pragma once

#include <SDL.h>

/* the short form wtk stands for
 * Windowing Tool Kit */
namespace wtk
{

class Widget {
    private:
        SDL_Window*     mWindow;
        SDL_GLContext   mContext;
        bool            mClose;

        Widget();

    public:
        Widget(int pWidth, int pHeight, const char* pTitle, const Widget* pWindow, const bool invisible);

        ~Widget();

        SDL_Window* getNativeHandle() const;

        void makeContextCurrent() const;

        long long getGLContextHandle();

        long long getDisplayHandle();

        void getFrameBufferSize(int* pW, int* pH);

        bool getClose() const;

        void setTitle(const char* pTitle);

        void setPos(int pX, int pY);

        void setClose(bool pClose);

        void swapBuffers();

        void hide();

        void show();

        bool close();

        void pollEvents();

};

}

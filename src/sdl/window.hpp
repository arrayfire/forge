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

#include <glm/glm.hpp>

/* the short form wtk stands for
 * Windowing Tool Kit */
namespace wtk
{

class Widget {
    private:
        SDL_Window*   mWindow;
        SDL_GLContext mContext;
        bool          mClose;
        uint32_t      mWindowId;
        float         mLastXPos;
        float         mLastYPos;
        glm::mat4     mMVP;
        int           mButton;
        SDL_Keycode   mMod;
        glm::vec3     mLastPos;

        Widget();

    public:
        Widget(int pWidth, int pHeight, const char* pTitle, const Widget* pWindow, const bool invisible);

        ~Widget();

        SDL_Window* getNativeHandle() const;

        void makeContextCurrent() const;

        long long getGLContextHandle();

        long long getDisplayHandle();

        void getFrameBufferSize(int* pW, int* pH);

        inline const glm::mat4& getMVP() const {
            return mMVP;
        }

        bool getClose() const;

        void setTitle(const char* pTitle);

        void setPos(int pX, int pY);

        void setSize(unsigned pW, unsigned pH);

        void setClose(bool pClose);

        void swapBuffers();

        void hide();

        void show();

        bool close();

        void resetCloseFlag();

        void pollEvents();

};

}

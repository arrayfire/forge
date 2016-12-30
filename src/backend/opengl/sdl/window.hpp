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
namespace forge
{
namespace wtk
{

void initWindowToolkit();
void destroyWindowToolkit();

class Widget {
    private:
        SDL_Window*   mWindow;
        SDL_GLContext mContext;
        bool          mClose;
        uint32_t      mWindowId;
        float         mLastXPos;
        float         mLastYPos;
        int           mButton;
        SDL_Keycode   mMod;
        glm::vec3     mLastPos;

        Widget();

        inline void getViewIds(int* pRow, int* pCol) {
            *pRow = mLastXPos/mCellWidth;
            *pCol = mLastYPos/mCellHeight;
        }

    public:
        /* public variables */
        int mWidth;     // Framebuffer width
        int mHeight;    // Framebuffer height
        int mRows;
        int mCols;
        int mCellWidth;
        int mCellHeight;
        std::vector<glm::mat4> mViewMatrices;
        std::vector<glm::mat4> mOrientMatrices;

        uint  mFramePBO;

        /* Constructors and methods */
        Widget(int pWidth, int pHeight, const char* pTitle, const Widget* pWindow, const bool invisible);

        ~Widget();

        SDL_Window* getNativeHandle() const;

        void makeContextCurrent() const;

        long long getGLContextHandle();

        long long getDisplayHandle();

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

        void resizePixelBuffers();
};

}
}

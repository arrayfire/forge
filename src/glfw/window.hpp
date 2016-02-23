/*******************************************************
* Copyright (c) 2015-2019, ArrayFire
* All rights reserved.
*
* This file is distributed under 3-clause BSD license.
* The complete license agreement can be obtained at:
* http://arrayfire.com/licenses/BSD-3-Clause
********************************************************/

#pragma once

#ifdef OS_WIN
    #define GLFW_EXPOSE_NATIVE_WIN32
    #define GLFW_EXPOSE_NATIVE_WGL
#endif

#ifdef OS_LNX
    #define GLFW_EXPOSE_NATIVE_X11
    #define GLFW_EXPOSE_NATIVE_GLX
#endif

#include <GLFW/glfw3.h>
#ifndef OS_MAC
#include <GLFW/glfw3native.h>
#endif

#include <glm/glm.hpp>

/* the short form wtk stands for
 * Windowing Tool Kit */
namespace wtk
{

class Widget {
    private:
        GLFWwindow* mWindow;
        bool        mClose;
        float       mLastXPos;
        float       mLastYPos;
        int         mButton;
        glm::vec3   mLastPos;

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
        std::vector<glm::mat4> mMVPs;

        uint    mIndex;
        GLuint  mFramePBOS[2];

        /* Constructors and methods */
        Widget(int pWidth, int pHeight, const char* pTitle, const Widget* pWindow, const bool invisible);

        ~Widget();

        GLFWwindow* getNativeHandle() const;

        void makeContextCurrent() const;

        long long getGLContextHandle();

        long long getDisplayHandle();

        void setTitle(const char* pTitle);

        void setPos(int pX, int pY);

        void setSize(unsigned pW, unsigned pH);

        void swapBuffers();

        void hide();

        void show();

        bool close();

        void resetCloseFlag();

        void resizeHandler(int pWidth, int pHeight);

        void keyboardHandler(int pKey, int pScancode, int pAction, int pMods);

        void cursorHandler(float pXPos, float pYPos);

        void mouseButtonHandler(int pButton, int pAction, int pMods);

        void pollEvents();

        void resizePixelBuffers();
};

}

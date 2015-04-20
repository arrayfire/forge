/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <fg/defines.h>

namespace fg
{

class FGAPI Window {
    private:
        ContextHandle   mCxt;
        DisplayHandle   mDsp;
        int             mID;

        int             mWidth;
        int             mHeight;
        GLFWwindow*     mWindow;
        /* single context for all windows */
        GLEWContext* mGLEWContext;
    protected:
        Window() {}

    public:
        Window(int pWidth, int pHeight, const char* pTitle, const Window* pWindow=NULL);
        ~Window();

        const ContextHandle context() const { return mCxt; }
        const DisplayHandle display() const { return mDsp; }
        int width() const { return mWidth; }
        int height() const { return mHeight; }
        GLFWwindow* window() const { return mWindow; }
        GLEWContext* glewContext() { return mGLEWContext; }
};

FGAPI void makeWindowCurrent(Window* pWindow);

}

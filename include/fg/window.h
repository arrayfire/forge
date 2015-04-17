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
        static GLEWContext* mGLEWContext;

    public:
        Window();
        Window(int pWidth, int pHeight, const char* pTitle);
        ~Window();

        const ContextHandle context() const;
        const DisplayHandle display() const;
        int width() const;
        int height() const;
        GLFWwindow* window() const;

        static void setGLEWcontext(GLEWContext* pCxt) { mGLEWContext = pCxt; }
        static GLEWContext* glewContext() { return mGLEWContext; }
};

FGAPI void makeWindowCurrent(Window* pWindow);

}

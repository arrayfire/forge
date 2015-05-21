/*******************************************************
* Copyright (c) 2015-2019, ArrayFire
* All rights reserved.
*
* This file is distributed under 3-clause BSD license.
* The complete license agreement can be obtained at:
* http://arrayfire.com/licenses/BSD-3-Clause
********************************************************/

#pragma once

#include <common.hpp>

#include <font.hpp>
#include <image.hpp>
#include <plot.hpp>
#include <histogram.hpp>

namespace internal
{

enum Renderable {
    FG_IMAGE = 1,
    FG_PLOT = 2,
    FG_HIST = 3
};

class _Window {
    private:
        ContextHandle mCxt;
        DisplayHandle mDsp;
        int           mID;

        int           mWidth;
        int           mHeight;
        GLFWwindow*   mWindow;
        _Font*         mFont;
        int           mRows;
        int           mCols;
        int           mCellWidth;
        int           mCellHeight;

        /* single context for all windows */
        GLEWContext* mGLEWContext;

    protected:
        _Window() {}

    public:
        _Window(int pWidth, int pHeight, const char* pTitle,
            const _Window* pWindow = NULL, const bool invisible = false);
        ~_Window();

        void setFont(internal::_Font* pFont) { mFont = pFont; }
        void setTitle(const char* pTitle);
        void setPos(int pX, int pY);

        void keyboardHandler(int pKey, int scancode, int pAction, int pMods);

        ContextHandle context() const { return mCxt; }
        DisplayHandle display() const { return mDsp; }
        int width() const { return mWidth; }
        int height() const { return mHeight; }
        GLFWwindow* window() const { return mWindow; }
        GLEWContext* glewContext() const { return mGLEWContext; }

        void hide() { glfwHideWindow(mWindow); }
        void show() { glfwShowWindow(mWindow); }
        bool close() { return glfwWindowShouldClose(mWindow)!=0; }

        /* draw functions */
        void draw(const internal::_Image* pImage);
        void draw(const internal::_Plot* pPlot);
        void draw(const internal::_Histogram* pHist);

        /* if the window render area is used to display
        * multiple Forge objects such as Image, Histogram, Plot etc
        * the following functions have to be used */
        void grid(int pRows, int pCols);
        /* below draw call uses zero-based indexing
        * for referring to cells within the grid */
        void draw(int pColId, int pRowId,
            const void* pRenderablePtr, Renderable pType,
            const char* pTitle = NULL);
        void draw();
};

}

void MakeContextCurrent(internal::_Window* pWindow);
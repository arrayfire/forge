/*******************************************************
* Copyright (c) 2015-2019, ArrayFire
* All rights reserved.
*
* This file is distributed under 3-clause BSD license.
* The complete license agreement can be obtained at:
* http://arrayfire.com/licenses/BSD-3-Clause
********************************************************/

#pragma once

#include <fg/font.h>
#include <fg/image.h>
#include <fg/plot.h>
#include <fg/histogram.h>

#include <common.hpp>
#include <font.hpp>
#include <image.hpp>
#include <plot.hpp>
#include <histogram.hpp>

#include <memory>

namespace internal
{

struct window_impl {
    ContextHandle mCxt;
    DisplayHandle mDsp;
    int           mID;
    int           mWidth;
    int           mHeight;
    GLFWwindow*   mWindow;
    int           mRows;
    int           mCols;
    int           mCellWidth;
    int           mCellHeight;
    GLEWContext*  mGLEWContext;
    fg::Font*     mFont;

    window_impl(int pWidth, int pHeight, const char* pTitle)
        : mWidth(pWidth), mHeight(pHeight), mWindow(nullptr),
        mRows(0), mCols(0) {
    }
    ~window_impl() {
        glfwDestroyWindow(mWindow);
    }

    inline void setGrid(int rows, int cols) {
        mRows = rows;
        mCols = cols;
    }

    inline void setCellDims(int w, int h) {
        mCellWidth = w;
        mCellHeight = h;
    }

    inline int rows() const { return mRows; }
    inline int cols() const { return mCols; }
    inline int cellw() const { return mCellWidth; }
    inline int cellh() const { return mCellHeight; }
};

class _Window {
    private:
        _Window() {}

    public:
        std::shared_ptr<window_impl> wnd;

        _Window(int pWidth, int pHeight, const char* pTitle,
            std::weak_ptr<_Window> pWindow, const bool invisible = false);

        void setFont(fg::Font* pFont) { wnd->mFont = pFont; }
        void setTitle(const char* pTitle);
        void setPos(int pX, int pY);

        void keyboardHandler(int pKey, int scancode, int pAction, int pMods);

        ContextHandle context() const { return wnd->mCxt; }
        DisplayHandle display() const { return wnd->mDsp; }
        int width() const { return wnd->mWidth; }
        int height() const { return wnd->mHeight; }
        GLEWContext* glewContext() const { return wnd->mGLEWContext; }
        GLFWwindow* get() const { return wnd->mWindow; }

        void hide() { glfwHideWindow(wnd->mWindow); }
        void show() { glfwShowWindow(wnd->mWindow); }
        bool close() { return glfwWindowShouldClose(wnd->mWindow) != 0; }

        /* draw functions */
        void draw(const fg::Image& pImage);
        void draw(const fg::Plot& pPlot);
        void draw(const fg::Histogram& pHist);

        /* if the window render area is used to display
        * multiple Forge objects such as Image, Histogram, Plot etc
        * the following functions have to be used */
        void grid(int pRows, int pCols);
        /* below draw call uses zero-based indexing
        * for referring to cells within the grid */
        template<typename T>
        void draw(int pColId, int pRowId, const T& pRenderable, const char* pTitle = NULL);

        void draw();
};

}

void MakeContextCurrent(internal::_Window* pWindow);

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

class window_impl {
    private:
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

        std::shared_ptr<font_impl>     mFont;

    public:
        window_impl(int pWidth, int pHeight, const char* pTitle,
                std::weak_ptr<window_impl> pWindow, const bool invisible=false);

        ~window_impl();

        void setFont(const std::shared_ptr<font_impl>& pFont);
        void setTitle(const char* pTitle);
        void setPos(int pX, int pY);

        void keyboardHandler(int pKey, int scancode, int pAction, int pMods);

        ContextHandle context() const;
        DisplayHandle display() const;
        int width() const;
        int height() const;
        GLEWContext* glewContext() const;
        GLFWwindow* get() const;

        void hide();
        void show();
        bool close();

        void draw(const std::shared_ptr<AbstractRenderable>& pRenderable);

        void grid(int pRows, int pCols);

        void draw(int pColId, int pRowId,
                const std::shared_ptr<AbstractRenderable>& pRenderable,
                const char* pTitle = NULL);

        void draw();
};

void MakeContextCurrent(window_impl* pWindow);

class _Window {
    private:
        std::shared_ptr<window_impl> wnd;

        _Window() {}

    public:

        _Window(int pWidth, int pHeight, const char* pTitle,
                const _Window* pWindow, const bool invisible = false) {
            if (pWindow) {
                wnd = std::make_shared<window_impl>(pWidth, pHeight, pTitle,
                                                    pWindow->impl(), invisible);
            } else {
                std::shared_ptr<window_impl> other;
                wnd = std::make_shared<window_impl>(pWidth, pHeight, pTitle,
                                                    other, invisible);
            }
        }

        inline const std::shared_ptr<window_impl>& impl () const {
            return wnd;
        }

        inline void setFont (_Font* pFont) {
            wnd->setFont (pFont->impl());
        }

        inline void setTitle(const char* pTitle) {
            wnd->setTitle(pTitle);
        }

        inline void setPos(int pX, int pY) {
            wnd->setPos(pX, pY);
        }

        inline ContextHandle context() const {
            return wnd->context() ;
        }

        inline DisplayHandle display() const {
            return wnd->display();
        }

        inline int width() const {
            return wnd->width();
        }

        inline int height() const {
            return wnd->height();
        }

        inline void makeCurrent() {
            MakeContextCurrent(wnd.get());
        }

        inline void hide() {
            wnd->hide();
        }

        inline void show() {
            wnd->show();
        }

        inline bool close() {
            return wnd->close();
        }

        inline void draw(const _Image* pImage) {
            wnd->draw(pImage->impl()) ;
        }

        inline void draw(const _Plot* pPlot) {
            wnd->draw(pPlot->impl()) ;
        }

        inline void draw(const _Histogram* pHist) {
            wnd->draw(pHist->impl()) ;
        }

        inline void draw() {
            wnd->draw();
        }

        inline void grid(int pRows, int pCols) {
            wnd->grid(pRows, pCols);
        }

        template<typename T>
        void draw(int pColId, int pRowId, const T* pRenderable, const char* pTitle = NULL) {
            wnd->draw(pColId, pRowId, pRenderable->impl(), pTitle);
        }
};

}

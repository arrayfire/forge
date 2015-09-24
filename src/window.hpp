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

#if defined(USE_GLFW)
#include <glfw/window.hpp>
#elif defined(USE_SDL)
#include <sdl/window.hpp>
#endif

#include <colormap.hpp>
#include <font.hpp>
#include <image.hpp>
#include <plot.hpp>
#include <plot3.hpp>
#include <histogram.hpp>

#include <memory>

namespace internal
{

class window_impl {
    private:
        long long     mCxt;
        long long     mDsp;
        int           mID;
        int           mWidth;
        int           mHeight;
        wtk::Widget*  mWindow;
        int           mRows;
        int           mCols;
        int           mCellWidth;
        int           mCellHeight;
        GLEWContext*  mGLEWContext;

        std::shared_ptr<font_impl>     mFont;
        std::shared_ptr<colormap_impl> mCMap;

        GLuint        mColorMapUBO;
        GLuint        mUBOSize;

    public:
        window_impl(int pWidth, int pHeight, const char* pTitle,
                std::weak_ptr<window_impl> pWindow, const bool invisible=false);

        ~window_impl();

        void setFont(const std::shared_ptr<font_impl>& pFont);
        void setTitle(const char* pTitle);
        void setPos(int pX, int pY);
        void setSize(unsigned pWidth, unsigned pHeight);
        void setColorMap(fg::ColorMap cmap);

        long long context() const;
        long long display() const;
        int width() const;
        int height() const;
        GLEWContext* glewContext() const;
        const wtk::Widget* get() const;
        const std::shared_ptr<colormap_impl>& colorMapPtr() const;

        void hide();
        void show();
        bool close();

        void draw(const std::shared_ptr<AbstractRenderable>& pRenderable);

        void grid(int pRows, int pCols);

        void draw(int pColId, int pRowId,
                  const std::shared_ptr<AbstractRenderable>& pRenderable,
                  const char* pTitle);

        void draw();
};

void MakeContextCurrent(const window_impl* pWindow);

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

        inline void setSize(unsigned pWidth, int pHeight) {
            wnd->setSize(pWidth, pHeight);
        }

        inline void setColorMap(fg::ColorMap cmap) {
            wnd->setColorMap(cmap);
        }

        inline long long context() const {
            return wnd->context() ;
        }

        inline long long display() const {
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

        inline void draw(_Image* pImage, const bool pKeepAspectRatio) {
            pImage->keepAspectRatio(pKeepAspectRatio);
            wnd->draw(pImage->impl()) ;
        }

        inline void draw(const _Plot* pPlot) {
            wnd->draw(pPlot->impl()) ;
        }

        inline void draw(const _Plot3* pPlot) {
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
        void draw(int pColId, int pRowId, T* pRenderable, const char* pTitle) {
            wnd->draw(pColId, pRowId, pRenderable->impl(), pTitle);
        }

        void draw(int pColId, int pRowId, _Image* pRenderable, const char* pTitle, const bool pKeepAspectRatio) {
            pRenderable->keepAspectRatio(pKeepAspectRatio);
            wnd->draw(pColId, pRowId, pRenderable->impl(), pTitle);
        }
};

}

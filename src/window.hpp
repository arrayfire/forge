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
#include <chart.hpp>

#include <memory>

namespace internal
{

class window_impl {
    private:
        long long     mCxt;
        long long     mDsp;
        int           mID;
        wtk::Widget*  mWindow;
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

        int getID() const;
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

        void swapBuffers();

        void saveFrameBuffer(const char* pFullPath);
};

void MakeContextCurrent(const window_impl* pWindow);

class _Window {
    private:
        std::shared_ptr<window_impl> mWindow;

        _Window() {}

    public:

        _Window(int pWidth, int pHeight, const char* pTitle,
                const _Window* pWindow, const bool invisible = false) {
            if (pWindow) {
                mWindow = std::make_shared<window_impl>(pWidth, pHeight, pTitle,
                                                    pWindow->impl(), invisible);
            } else {
                std::shared_ptr<window_impl> other;
                mWindow = std::make_shared<window_impl>(pWidth, pHeight, pTitle,
                                                    other, invisible);
            }
        }

        inline const std::shared_ptr<window_impl>& impl () const {
            return mWindow;
        }

        inline void setFont (_Font* pFont) {
            mWindow->setFont (pFont->impl());
        }

        inline void setTitle(const char* pTitle) {
            mWindow->setTitle(pTitle);
        }

        inline void setPos(int pX, int pY) {
            mWindow->setPos(pX, pY);
        }

        inline void setSize(unsigned pWidth, int pHeight) {
            mWindow->setSize(pWidth, pHeight);
        }

        inline void setColorMap(fg::ColorMap cmap) {
            mWindow->setColorMap(cmap);
        }

        inline int getID() const {
            return mWindow->getID();
        }

        inline long long context() const {
            return mWindow->context() ;
        }

        inline long long display() const {
            return mWindow->display();
        }

        inline int width() const {
            return mWindow->width();
        }

        inline int height() const {
            return mWindow->height();
        }

        inline void makeCurrent() {
            MakeContextCurrent(mWindow.get());
        }

        inline void hide() {
            mWindow->hide();
        }

        inline void show() {
            mWindow->show();
        }

        inline bool close() {
            return mWindow->close();
        }

        inline void draw(_Image* pImage, const bool pKeepAspectRatio) {
            pImage->keepAspectRatio(pKeepAspectRatio);
            mWindow->draw(pImage->impl()) ;
        }

        inline void draw(const _Chart* pChart) {
            mWindow->draw(pChart->impl()) ;
        }

        inline void swapBuffers() {
            mWindow->swapBuffers();
        }

        inline void grid(int pRows, int pCols) {
            mWindow->grid(pRows, pCols);
        }

        template<typename T>
        void draw(int pColId, int pRowId, T* pRenderable, const char* pTitle) {
            mWindow->draw(pColId, pRowId, pRenderable->impl(), pTitle);
        }

        void draw(int pColId, int pRowId, _Image* pRenderable, const char* pTitle, const bool pKeepAspectRatio) {
            pRenderable->keepAspectRatio(pKeepAspectRatio);
            mWindow->draw(pColId, pRowId, pRenderable->impl(), pTitle);
        }

        inline void saveFrameBuffer(const char* pFullPath) {
            mWindow->saveFrameBuffer(pFullPath);
        }
};

}

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

#include <colormap_impl.hpp>
#include <font_impl.hpp>
#include <image_impl.hpp>
#include <chart_impl.hpp>

#include <memory>

namespace forge
{
namespace opengl
{

class window_impl {
    private:
        long long     mCxt;
        long long     mDsp;
        int           mID;
        wtk::Widget*  mWindow;

        std::shared_ptr<font_impl>     mFont;
        std::shared_ptr<colormap_impl> mCMap;

        gl::GLuint mColorMapUBO;
        gl::GLuint mUBOSize;

    public:
        window_impl(int pWidth, int pHeight, const char* pTitle,
                std::weak_ptr<window_impl> pWindow, const bool invisible=false);

        ~window_impl();

        void setFont(const std::shared_ptr<font_impl>& pFont);
        void setTitle(const char* pTitle);
        void setPos(int pX, int pY);
        void setSize(unsigned pWidth, unsigned pHeight);
        void setColorMap(forge::ColorMap cmap);

        int getID() const;
        long long context() const;
        long long display() const;
        int width() const;
        int height() const;
        const wtk::Widget* get() const;
        const std::shared_ptr<colormap_impl>& colorMapPtr() const;

        void hide();
        void show();
        bool close();

        void draw(const std::shared_ptr<AbstractRenderable>& pRenderable);

        void grid(int pRows, int pCols);

        void getGrid(int *pRows, int *pCols);

        void draw(int pColId, int pRowId,
                  const std::shared_ptr<AbstractRenderable>& pRenderable,
                  const char* pTitle);

        void swapBuffers();

        void saveFrameBuffer(const char* pFullPath);
};

void MakeContextCurrent(const window_impl* pWindow);

}
}

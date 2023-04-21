/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <window.hpp>

#include <chart_impl.hpp>
#include <colormap_impl.hpp>
#include <common/defines.hpp>
#include <font_impl.hpp>
#include <image_impl.hpp>
#include <plot_impl.hpp>

#include <memory>

namespace forge {
namespace opengl {

class window_impl {
   private:
    long long mCxt;
    long long mDsp;
    int mID;
    std::unique_ptr<wtk::Widget> mWidget;

    std::shared_ptr<font_impl> mFont;
    std::shared_ptr<colormap_impl> mCMap;
    std::shared_ptr<plot_impl> mArcBallLoop0;
    std::shared_ptr<plot_impl> mArcBallLoop1;

    uint32_t mColorMapUBO;
    uint32_t mUBOSize;

    void prepArcBallObjects();

   public:
    window_impl(int pWidth, int pHeight, const char* pTitle,
                std::weak_ptr<window_impl> pWindow,
                const bool invisible = false);

    ~window_impl();

    void makeContextCurrent();
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
    const std::unique_ptr<wtk::Widget>& get() const;
    const std::shared_ptr<colormap_impl>& colorMapPtr() const;

    void hide();
    void show();
    bool close();

    void draw(const std::shared_ptr<AbstractRenderable>& pRenderable);

    void draw(const int pRows, const int pCols, const int pIndex,
              const std::shared_ptr<AbstractRenderable>& pRenderable,
              const char* pTitle);

    void swapBuffers();

    void saveFrameBuffer(const char* pFullPath);
};

}  // namespace opengl
}  // namespace forge

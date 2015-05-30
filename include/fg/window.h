/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <fg/defines.h>
#include <fg/font.h>
#include <fg/image.h>
#include <fg/plot.h>
#include <fg/histogram.h>

namespace internal
{
class _Window;
}

namespace fg
{

class Window {
    private:
        internal::_Window* value;

        Window() {}

    public:
        FGAPI Window(int pWidth, int pHeight, const char* pTitle,
                    const Window* pWindow=NULL, const bool invisible = false);
        FGAPI Window(const Window& other);
        FGAPI ~Window();

        FGAPI void setFont(Font* pFont);
        FGAPI void setTitle(const char* pTitle);
        FGAPI void setPos(int pX, int pY);
        FGAPI void setColorMap(ColorMap cmap);

        FGAPI ContextHandle context() const;
        FGAPI DisplayHandle display() const;
        FGAPI int width() const;
        FGAPI int height() const;
        FGAPI internal::_Window* get() const;

        FGAPI void makeCurrent();
        FGAPI void hide();
        FGAPI void show();
        FGAPI bool close();

        /* draw functions */
        FGAPI void draw(const Image& pImage);
        FGAPI void draw(const Plot& pPlot);
        FGAPI void draw(const Histogram& pHist);

        /* if the window render area is used to display
         * multiple Forge objects such as Image, Histogram, Plot etc
         * the following functions have to be used */
        FGAPI void grid(int pRows, int pCols);

        /* below draw call uses zero-based indexing
         * for referring to cells within the grid */
        FGAPI void draw(int pColId, int pRowId, const Image& pImage, const char* pTitle=NULL);
        FGAPI void draw(int pColId, int pRowId, const Plot& pPlot, const char* pTitle = NULL);
        FGAPI void draw(int pColId, int pRowId, const Histogram& pHist, const char* pTitle = NULL);

        FGAPI void draw();
};

}

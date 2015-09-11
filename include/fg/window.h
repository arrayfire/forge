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

/**
   \class Window

   \brief Window is where other objects such as Images, Plots etc. are rendered.
 */
class Window {
    private:
        internal::_Window* value;

        Window() {}

    public:
        /**
           Creates a Window object.

           \param[in] pWidth Width of the display window
           \param[in] pHeight Height of the display window
           \param[in] pTitle window Title
           \param[in] pWindow An already existing window with which the window to
                      be created should share the OpenGL context.
           \param[in] invisible window is created in invisible mode.
                      User has to call Window::show() when they decide
                      to actually display the window
         */
        FGAPI Window(int pWidth, int pHeight, const char* pTitle,
                    const Window* pWindow=0, const bool invisible = false);

        /**
           Copy constructor for Window

           \param[in] other is the Window of which we make a copy of.
         */
        FGAPI Window(const Window& other);

        /**
           Window Destructor
         */
        FGAPI ~Window();

        /**
           Set font to be used by the window to draw text

           \param[in] pFont Font object pointer
         */
        FGAPI void setFont(Font* pFont);

        /**
           Set the window title

           \param[in] pTitle is the window title
         */
        FGAPI void setTitle(const char* pTitle);

        /**
           Set the start position where the window will appear

           \param[in] pX is horizontal coordinate
           \param[in] pY is vertical coordinate
         */
        FGAPI void setPos(int pX, int pY);

        /**
           Set the size of the window programmatically

           \param[in] pWidth target width
           \param[in] pHeight target height
         */
        FGAPI void setSize(unsigned pWidth, unsigned pHeight);

        /**
           Set the colormap to be used for subsequent rendering calls

           \param[in] cmap should be one of the enum values from \ref ColorMap
         */
        FGAPI void setColorMap(ColorMap cmap);

        /**
           Get OpenGL context handle
           \return Context handle for the window's OpenGL context
         */
        FGAPI long long context() const;

        /**
           Get Native Window display handle
           \return Display handle of the native window implemenation of Window
         */
        FGAPI long long display() const;

        /**
           \return window width
         */
        FGAPI int width() const;

        /**
           \return window height
         */
        FGAPI int height() const;

        /**
           \return internal handle for window implementation
         */
        FGAPI internal::_Window* get() const;

        /**
           Make the current window's OpenGL context active context
         */
        FGAPI void makeCurrent();

        /**
           Hide the window
         */
        FGAPI void hide();

        /**
           Show the window if hidden, otherwise no effect
         */
        FGAPI void show();

        /**
           Check if the window is ready for close. This happens when an user
           presses `ESC` key while the window is in focus or clicks on the close
           button of the window

           \return true | false
         */
        FGAPI bool close();

        /**
           Render an Image to Window

           \param[in] pImage is an object of class Image

           \note this draw call does a OpenGL swap buffer, so we do not need
           to call Window::draw() after this function is called upon for rendering
           an image
         */
        FGAPI void draw(const Image& pImage, const bool pKeepAspectRatio=true);

        /**
           Render a Plot to Window

           \param[in] pPlot is an object of class Plot

           \note this draw call does a OpenGL swap buffer, so we do not need
           to call Window::draw() after this function is called upon for rendering
           a plot
         */
        FGAPI void draw(const Plot& pPlot);

        /**
           Render Histogram to Window

           \param[in] pHist is an object of class Histogram

           \note this draw call does a OpenGL swap buffer, so we do not need
           to call Window::draw() after this function is called upon for rendering
           a histogram
         */
        FGAPI void draw(const Histogram& pHist);

        /**
           Setup grid layout for multivew mode

           Multiview mode is where you can render different objects
           to different sub-regions of a given window

           \param[in] pRows is number of rows in grid layout
           \param[in] pCols is number of coloumns in grid layout
         */
        FGAPI void grid(int pRows, int pCols);

        /**
           Render Image to given sub-region of the window in multiview mode

           Window::grid should have been already called before any of the draw calls
           that accept coloum index and row index is used to render an object.

           \param[in] pColId is coloumn index
           \param[in] pRowId is row index
           \param[in] pImage is an object of class Image
           \param[in] pTitle is the title that will be displayed for the cell represented
                      by \p pColId and \p pRowId

           \note This draw call doesn't do OpenGL swap buffer since it doesn't have the
           knowledge of which sub-regions already got rendered. We should call
           Window::draw() once all draw calls corresponding to all sub-regions are called
           when in multiview mode.
         */
        FGAPI void draw(int pColId, int pRowId, const Image& pImage, const char* pTitle=0, const bool pKeepAspectRatio=true);

        /**
           Render Plot to given sub-region of the window in multiview mode

           Window::grid should have been already called before any of the draw calls
           that accept coloum index and row index is used to render an object.

           \param[in] pColId is coloumn index
           \param[in] pRowId is row index
           \param[in] pPlot is an object of class Plot
           \param[in] pTitle is the title that will be displayed for the cell represented
                      by \p pColId and \p pRowId

           \note This draw call doesn't do OpenGL swap buffer since it doesn't have the
           knowledge of which sub-regions already got rendered. We should call
           Window::draw() once all draw calls corresponding to all sub-regions are called
           when in multiview mode.
         */
        FGAPI void draw(int pColId, int pRowId, const Plot& pPlot, const char* pTitle = 0);

        /**
           Render Histogram to given sub-region of the window in multiview mode

           Window::grid should have been already called before any of the draw calls
           that accept coloum index and row index is used to render an object.

           \param[in] pColId is coloumn index
           \param[in] pRowId is row index
           \param[in] pHist is an object of class Histogram
           \param[in] pTitle is the title that will be displayed for the cell represented
                      by \p pColId and \p pRowId

           \note This draw call doesn't do OpenGL swap buffer since it doesn't have the
           knowledge of which sub-regions already got rendered. We should call
           Window::draw() once all draw calls corresponding to all sub-regions are called
           when in multiview mode.
         */
        FGAPI void draw(int pColId, int pRowId, const Histogram& pHist, const char* pTitle = 0);

        /**
           Swaps background OpenGL buffer with front buffer

           This draw call should only be used when the window is displaying
           something in multiview mode
         */
        FGAPI void draw();
};

}

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
#include <fg/chart.h>
#include <fg/surface.h>
#include <fg/histogram.h>


#ifdef __cplusplus
extern "C" {
#endif

FGAPI fg_err fg_create_window(fg_window *pWindow,
                              const int pWidth, const int pHeight,
                              const char* pTitle,
                              const fg_window pShareWindow,
                              const bool pInvisible);

FGAPI fg_err fg_destroy_window(fg_window pWindow);

FGAPI fg_err fg_set_window_font(fg_window pWindow, fg_font pFont);

FGAPI fg_err fg_set_window_title(fg_window pWindow, const char* pTitle);

FGAPI fg_err fg_set_window_position(fg_window pWindow, const int pX, const int pY);

FGAPI fg_err fg_set_window_size(fg_window pWindow, const uint pWidth, const uint pHeight);

FGAPI fg_err fg_set_window_colormap(fg_window pWindow, const fg_color_map pColorMap);

FGAPI fg_err fg_get_window_context_handle(long long *pContext, const fg_window pWindow);

FGAPI fg_err fg_get_window_display_handle(long long *pDisplay, const fg_window pWindow);

FGAPI fg_err fg_get_window_width(int *pWidth, const fg_window pWindow);

FGAPI fg_err fg_get_window_height(int *pHeight, const fg_window pWindow);

FGAPI fg_err fg_make_window_current(const fg_window pWindow);

FGAPI fg_err fg_hide_window(const fg_window pWindow);

FGAPI fg_err fg_show_window(const fg_window pWindow);

FGAPI fg_err fg_close_window(bool* pIsClosed, const fg_window pWindow);

FGAPI fg_err fg_draw_image(const fg_window pWindow, const fg_image pImage, const bool pKeepAspectRatio);

FGAPI fg_err fg_draw_chart(const fg_window pWindow, const fg_chart pChart);

FGAPI fg_err fg_setup_window_layout(int pRows, int pCols, fg_window pWindow);

FGAPI fg_err fg_draw_image_to_cell(const fg_window pWindow, int pColId, int pRowId,
                                   const fg_image pImage, const char* pTitle, const bool pKeepAspectRatio);

FGAPI fg_err fg_draw_chart_to_cell(const fg_window pWindow, int pColId, int pRowId,
                                   const fg_chart pChart, const char* pTitle);

FGAPI fg_err fg_swap_window_buffers(const fg_window pWindow);

FGAPI fg_err fg_save_window_framebuffer(const char* pFullPath, const fg_window pWindow);

#ifdef __cplusplus
}
#endif


#ifdef __cplusplus

namespace fg
{

/**
   \class Window

   \brief Window is where other objects such as Images, Plots etc. are rendered.
 */
class Window {
    private:
        fg_window mValue;

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
        FGAPI fg_window get() const;

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
           \param[in] pKeepAspectRatio when set to true keeps the aspect ratio
                      of the input image constant.

           \note this draw call does a OpenGL swap buffer, so we do not need
           to call Window::draw() after this function is called upon for rendering
           an image
         */
        FGAPI void draw(const Image& pImage, const bool pKeepAspectRatio=true);

        /**
           Render a chart to Window

           \param[in] pChart is an chart object

           \note this draw call does a OpenGL swap buffer, so we do not need
           to call Window::draw() after this function is called upon for rendering
           a plot
         */
        FGAPI void draw(const Chart& pChart);

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
           \param[in] pKeepAspectRatio when set to true keeps the aspect ratio
                      of the input image constant.

           \note This draw call doesn't do OpenGL swap buffer since it doesn't have the
           knowledge of which sub-regions already got rendered. We should call
           Window::draw() once all draw calls corresponding to all sub-regions are called
           when in multiview mode.
         */
        FGAPI void draw(int pColId, int pRowId, const Image& pImage, const char* pTitle=0, const bool pKeepAspectRatio=true);

        /**
           Render the chart to given sub-region of the window in multiview mode

           Window::grid should have been already called before any of the draw calls
           that accept coloum index and row index is used to render an object.

           \param[in] pColId is coloumn index
           \param[in] pRowId is row index
           \param[in] pChart is a Chart with one or more plottable renderables
           \param[in] pTitle is the title that will be displayed for the cell represented
                      by \p pColId and \p pRowId

           \note This draw call doesn't do OpenGL swap buffer since it doesn't have the
           knowledge of which sub-regions already got rendered. We should call
           Window::draw() once all draw calls corresponding to all sub-regions are called
           when in multiview mode.
         */
        FGAPI void draw(int pColId, int pRowId, const Chart& pChart, const char* pTitle = 0);

        /**
           Swaps background OpenGL buffer with front buffer

           This draw call should only be used when the window is displaying
           something in multiview mode
         */
        FGAPI void swapBuffers();

        /**
           Save window frame buffer to give location in provided image format

           The image format to be saved in is inferred from the file extension
           provided in the path string.

           \param[in] pFullPath should be the absolute path of the target location
                      where the framebuffer should be stored. The target image format
                      is inferred from the file extension.
         */
        FGAPI void saveFrameBuffer(const char* pFullPath);
};

}

#endif

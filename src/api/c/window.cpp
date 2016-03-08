/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fg/window.h>

#include <handle.hpp>
#include <err_common.hpp>
#include <Window.hpp>

fg_err fg_create_window(fg_window *pWindow,
                        const int pWidth, const int pHeight,
                        const char* pTitle,
                        const fg_window pShareWindow,
                        const bool pInvisible)
{
    try {
        common::Window* shrdWnd = getWindow(pShareWindow);
        common::Window* temp = nullptr;
        if (shrdWnd == nullptr) {
            temp = new common::Window(pWidth, pHeight, pTitle, nullptr, pInvisible);
        } else {
            temp = new common::Window(pWidth, pHeight, pTitle, shrdWnd, pInvisible);
        }
        *pWindow = getHandle(temp);
    }
    CATCHALL

    return FG_SUCCESS;
}

fg_err fg_destroy_window(fg_window pWindow)
{
    try {
        delete getWindow(pWindow);
    }
    CATCHALL

    return FG_SUCCESS;
}

fg_err fg_set_window_font(fg_window pWindow, fg_font pFont)
{
    try {
        getWindow(pWindow)->setFont(getFont(pFont));
    }
    CATCHALL
    return FG_SUCCESS;
}

fg_err fg_set_window_title(fg_window pWindow, const char* pTitle)
{
    try {
        getWindow(pWindow)->setTitle(pTitle);
    }
    CATCHALL
    return FG_SUCCESS;
}

fg_err fg_set_window_position(fg_window pWindow, const int pX, const int pY)
{
    try {
        getWindow(pWindow)->setPos(pX, pY);
    }
    CATCHALL
    return FG_SUCCESS;
}

fg_err fg_set_window_size(fg_window pWindow, const uint pWidth, const uint pHeight)
{
    try {
        getWindow(pWindow)->setSize(pWidth, pHeight);
    }
    CATCHALL
    return FG_SUCCESS;
}

fg_err fg_set_window_colormap(fg_window pWindow, const fg_color_map pColorMap)
{
    try {
        getWindow(pWindow)->setColorMap(pColorMap);
    }
    CATCHALL
    return FG_SUCCESS;
}

fg_err fg_get_window_context_handle(long long *pContext, const fg_window pWindow)
{
    try {
        *pContext = getWindow(pWindow)->context();
    }
    CATCHALL
    return FG_SUCCESS;
}

fg_err fg_get_window_display_handle(long long *pDisplay, const fg_window pWindow)
{
    try {
        *pDisplay = getWindow(pWindow)->display();
    }
    CATCHALL
    return FG_SUCCESS;
}

fg_err fg_get_window_width(int *pWidth, const fg_window pWindow)
{
    try {
        *pWidth = getWindow(pWindow)->width();
    }
    CATCHALL
    return FG_SUCCESS;
}

fg_err fg_get_window_height(int *pHeight, const fg_window pWindow)
{
    try {
        *pHeight = getWindow(pWindow)->height();
    }
    CATCHALL
    return FG_SUCCESS;
}

fg_err fg_make_window_current(const fg_window pWindow)
{
    try {
        getWindow(pWindow)->makeCurrent();
    }
    CATCHALL
    return FG_SUCCESS;
}

fg_err fg_hide_window(const fg_window pWindow)
{
    try {
        getWindow(pWindow)->hide();
    }
    CATCHALL
    return FG_SUCCESS;
}

fg_err fg_show_window(const fg_window pWindow)
{
    try {
        getWindow(pWindow)->show();
    }
    CATCHALL
    return FG_SUCCESS;
}

fg_err fg_close_window(bool* pIsClosed, const fg_window pWindow)
{
    try {
        *pIsClosed = getWindow(pWindow)->close();
    }
    CATCHALL
    return FG_SUCCESS;
}

fg_err fg_draw_image(const fg_window pWindow, const fg_image pImage, const bool pKeepAspectRatio)
{
    try {
        getWindow(pWindow)->draw(getImage(pImage), pKeepAspectRatio);
    }
    CATCHALL
    return FG_SUCCESS;
}

fg_err fg_draw_chart(const fg_window pWindow, const fg_chart pChart)
{
    try {
        getWindow(pWindow)->draw(getChart(pChart));
    }
    CATCHALL
    return FG_SUCCESS;
}

fg_err fg_setup_window_layout(int pRows, int pCols, fg_window pWindow)
{
    try {
        getWindow(pWindow)->grid(pRows, pCols);
    }
    CATCHALL
    return FG_SUCCESS;
}

fg_err fg_draw_image_to_cell(const fg_window pWindow, int pColId, int pRowId,
                             const fg_image pImage, const char* pTitle, const bool pKeepAspectRatio)
{
    try {
        getWindow(pWindow)->draw(pColId, pRowId, getImage(pImage), pTitle, pKeepAspectRatio);
    }
    CATCHALL
    return FG_SUCCESS;
}

fg_err fg_draw_chart_to_cell(const fg_window pWindow, int pColId, int pRowId,
                             const fg_chart pChart, const char* pTitle)
{
    try {
        getWindow(pWindow)->draw(pColId, pRowId, getChart(pChart), pTitle);
    }
    CATCHALL
    return FG_SUCCESS;
}

fg_err fg_swap_window_buffers(const fg_window pWindow)
{
    try {
        getWindow(pWindow)->swapBuffers();
    }
    CATCHALL
    return FG_SUCCESS;
}

fg_err fg_save_window_framebuffer(const char* pFullPath, const fg_window pWindow)
{
    try {
        getWindow(pWindow)->saveFrameBuffer(pFullPath);
    }
    CATCHALL
    return FG_SUCCESS;
}

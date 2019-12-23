/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/err_handling.hpp>
#include <common/handle.hpp>
#include <common/window.hpp>
#include <fg/window.h>

using namespace forge;

using forge::common::getChart;
using forge::common::getFont;
using forge::common::getImage;
using forge::common::getWindow;

fg_err fg_create_window(fg_window* pWindow, const int pWidth, const int pHeight,
                        const char* pTitle, const fg_window pShareWindow,
                        const bool pInvisible) {
    try {
        ARG_ASSERT(1, (pWidth > 0));
        ARG_ASSERT(2, (pHeight > 0));

        common::Window* shrdWnd = getWindow(pShareWindow);
        common::Window* temp    = nullptr;
        if (shrdWnd == nullptr) {
            temp = new common::Window(pWidth, pHeight, pTitle, nullptr,
                                      pInvisible);
        } else {
            temp = new common::Window(pWidth, pHeight, pTitle, shrdWnd,
                                      pInvisible);
        }
        *pWindow = getHandle(temp);
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_retain_window(fg_window* pOut, fg_window pWindow) {
    try {
        ARG_ASSERT(1, (pWindow != 0));

        common::Window* temp = new common::Window(pWindow);
        *pOut                = getHandle(temp);
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_release_window(fg_window pWindow) {
    try {
        ARG_ASSERT(0, (pWindow != 0));

        delete getWindow(pWindow);
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_set_window_font(fg_window pWindow, const fg_font pFont) {
    try {
        ARG_ASSERT(0, (pWindow != 0));
        ARG_ASSERT(1, (pFont != 0));

        getWindow(pWindow)->setFont(getFont(pFont));
    }
    CATCHALL
    return FG_ERR_NONE;
}

fg_err fg_set_window_title(fg_window pWindow, const char* pTitle) {
    try {
        ARG_ASSERT(0, (pWindow != 0));
        ARG_ASSERT(1, (pTitle != 0));

        getWindow(pWindow)->setTitle(pTitle);
    }
    CATCHALL
    return FG_ERR_NONE;
}

fg_err fg_set_window_position(fg_window pWindow, const int pX, const int pY) {
    try {
        ARG_ASSERT(0, (pWindow != 0));
        ARG_ASSERT(1, (pX >= 0));
        ARG_ASSERT(2, (pY >= 0));

        getWindow(pWindow)->setPos(pX, pY);
    }
    CATCHALL
    return FG_ERR_NONE;
}

fg_err fg_set_window_size(fg_window pWindow, const unsigned pWidth,
                          const unsigned pHeight) {
    try {
        ARG_ASSERT(0, (pWindow != 0));
        ARG_ASSERT(1, (pWidth > 0));
        ARG_ASSERT(2, (pHeight > 0));

        getWindow(pWindow)->setSize(pWidth, pHeight);
    }
    CATCHALL
    return FG_ERR_NONE;
}

fg_err fg_set_window_colormap(fg_window pWindow, const fg_color_map pColorMap) {
    try {
        ARG_ASSERT(0, (pWindow != 0));

        getWindow(pWindow)->setColorMap(pColorMap);
    }
    CATCHALL
    return FG_ERR_NONE;
}

fg_err fg_get_window_context_handle(long long* pContext,
                                    const fg_window pWindow) {
    try {
        ARG_ASSERT(1, (pWindow != 0));

        *pContext = getWindow(pWindow)->context();
    }
    CATCHALL
    return FG_ERR_NONE;
}

fg_err fg_get_window_display_handle(long long* pDisplay,
                                    const fg_window pWindow) {
    try {
        ARG_ASSERT(1, (pWindow != 0));

        *pDisplay = getWindow(pWindow)->display();
    }
    CATCHALL
    return FG_ERR_NONE;
}

fg_err fg_get_window_width(int* pWidth, const fg_window pWindow) {
    try {
        ARG_ASSERT(1, (pWindow != 0));

        *pWidth = getWindow(pWindow)->width();
    }
    CATCHALL
    return FG_ERR_NONE;
}

fg_err fg_get_window_height(int* pHeight, const fg_window pWindow) {
    try {
        ARG_ASSERT(1, (pWindow != 0));

        *pHeight = getWindow(pWindow)->height();
    }
    CATCHALL
    return FG_ERR_NONE;
}

fg_err fg_make_window_current(const fg_window pWindow) {
    try {
        ARG_ASSERT(0, (pWindow != 0));

        getWindow(pWindow)->makeCurrent();
    }
    CATCHALL
    return FG_ERR_NONE;
}

fg_err fg_hide_window(const fg_window pWindow) {
    try {
        ARG_ASSERT(0, (pWindow != 0));

        getWindow(pWindow)->hide();
    }
    CATCHALL
    return FG_ERR_NONE;
}

fg_err fg_show_window(const fg_window pWindow) {
    try {
        ARG_ASSERT(0, (pWindow != 0));

        getWindow(pWindow)->show();
    }
    CATCHALL
    return FG_ERR_NONE;
}

fg_err fg_close_window(bool* pIsClosed, const fg_window pWindow) {
    try {
        ARG_ASSERT(1, (pWindow != 0));

        *pIsClosed = getWindow(pWindow)->close();
    }
    CATCHALL
    return FG_ERR_NONE;
}

fg_err fg_draw_image(const fg_window pWindow, const fg_image pImage,
                     const bool pKeepAspectRatio) {
    try {
        ARG_ASSERT(0, (pWindow != 0));
        ARG_ASSERT(1, (pImage != 0));

        getWindow(pWindow)->draw(getImage(pImage), pKeepAspectRatio);
    }
    CATCHALL
    return FG_ERR_NONE;
}

fg_err fg_draw_chart(const fg_window pWindow, const fg_chart pChart) {
    try {
        ARG_ASSERT(0, (pWindow != 0));
        ARG_ASSERT(1, (pChart != 0));

        getWindow(pWindow)->draw(getChart(pChart));
    }
    CATCHALL
    return FG_ERR_NONE;
}

fg_err fg_draw_image_to_cell(const fg_window pWindow, const int pRows,
                             const int pCols, const int pIndex,
                             const fg_image pImage, const char* pTitle,
                             const bool pKeepAspectRatio) {
    try {
        ARG_ASSERT(0, (pWindow != 0));
        ARG_ASSERT(1, (pRows > 0));
        ARG_ASSERT(2, (pCols > 0));
        ARG_ASSERT(3, (pIndex >= 0));
        ARG_ASSERT(4, (pImage != 0));

        getWindow(pWindow)->draw(pRows, pCols, pIndex, getImage(pImage), pTitle,
                                 pKeepAspectRatio);
    }
    CATCHALL
    return FG_ERR_NONE;
}

fg_err fg_draw_chart_to_cell(const fg_window pWindow, const int pRows,
                             const int pCols, const int pIndex,
                             const fg_chart pChart, const char* pTitle) {
    try {
        ARG_ASSERT(0, (pWindow != 0));
        ARG_ASSERT(1, (pRows > 0));
        ARG_ASSERT(2, (pCols > 0));
        ARG_ASSERT(3, (pIndex >= 0));
        ARG_ASSERT(4, (pChart != 0));

        getWindow(pWindow)->draw(pRows, pCols, pIndex, getChart(pChart),
                                 pTitle);
    }
    CATCHALL
    return FG_ERR_NONE;
}

fg_err fg_swap_window_buffers(const fg_window pWindow) {
    try {
        ARG_ASSERT(0, (pWindow != 0));

        getWindow(pWindow)->swapBuffers();
    }
    CATCHALL
    return FG_ERR_NONE;
}

fg_err fg_save_window_framebuffer(const char* pFullPath,
                                  const fg_window pWindow) {
    try {
        ARG_ASSERT(0, pFullPath != NULL);
        ARG_ASSERT(1, (pWindow != 0));

        getWindow(pWindow)->saveFrameBuffer(pFullPath);
    }
    CATCHALL
    return FG_ERR_NONE;
}

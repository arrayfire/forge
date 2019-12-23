/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fg/window.h>

#include <error.hpp>

#include <utility>

namespace forge {
Window::Window(const int pWidth, const int pHeight, const char* pTitle,
               const Window* pWindow, const bool invisible)
    : mValue(0) {
    fg_window temp = 0;
    fg_window shrd = (pWindow ? pWindow->get() : 0);
    FG_THROW(fg_create_window(&temp, pWidth, pHeight, pTitle, shrd, invisible));

    std::swap(mValue, temp);
}

Window::Window(const Window& other) {
    fg_window temp = 0;

    FG_THROW(fg_retain_window(&temp, other.get()));

    std::swap(mValue, temp);
}

Window::~Window() { fg_release_window(get()); }

void Window::setFont(Font* pFont) {
    FG_THROW(fg_set_window_font(get(), pFont->get()));
}

void Window::setTitle(const char* pTitle) {
    FG_THROW(fg_set_window_title(get(), pTitle));
}

void Window::setPos(const int pX, const int pY) {
    FG_THROW(fg_set_window_position(get(), pX, pY));
}

void Window::setSize(const unsigned pW, const unsigned pH) {
    FG_THROW(fg_set_window_size(get(), pW, pH));
}

void Window::setColorMap(const ColorMap cmap) {
    FG_THROW(fg_set_window_colormap(get(), cmap));
}

long long Window::context() const {
    long long contextHandle = 0;
    FG_THROW(fg_get_window_context_handle(&contextHandle, get()));
    return contextHandle;
}

long long Window::display() const {
    long long displayHandle = 0;
    FG_THROW(fg_get_window_display_handle(&displayHandle, get()));
    return displayHandle;
}

int Window::width() const {
    int retVal = 0;
    FG_THROW(fg_get_window_width(&retVal, get()));
    return retVal;
}

int Window::height() const {
    int retVal = 0;
    FG_THROW(fg_get_window_height(&retVal, get()));
    return retVal;
}

fg_window Window::get() const { return mValue; }

void Window::makeCurrent() { FG_THROW(fg_make_window_current(get())); }

void Window::hide() { FG_THROW(fg_hide_window(get())); }

void Window::show() { FG_THROW(fg_show_window(get())); }

bool Window::close() {
    bool isClosed = false;
    FG_THROW(fg_close_window(&isClosed, get()));
    return isClosed;
}

void Window::draw(const Image& pImage, const bool pKeepAspectRatio) {
    FG_THROW(fg_draw_image(get(), pImage.get(), pKeepAspectRatio));
}

void Window::draw(const Chart& pChart) {
    FG_THROW(fg_draw_chart(get(), pChart.get()));
}

void Window::draw(const int pRows, const int pCols, const int pIndex,
                  const Image& pImage, const char* pTitle,
                  const bool pKeepAspectRatio) {
    FG_THROW(fg_draw_image_to_cell(get(), pRows, pCols, pIndex, pImage.get(),
                                   pTitle, pKeepAspectRatio));
}

void Window::draw(const int pRows, const int pCols, const int pIndex,
                  const Chart& pChart, const char* pTitle) {
    FG_THROW(fg_draw_chart_to_cell(get(), pRows, pCols, pIndex, pChart.get(),
                                   pTitle));
}

void Window::swapBuffers() { FG_THROW(fg_swap_window_buffers(get())); }

void Window::saveFrameBuffer(const char* pFullPath) {
    FG_THROW(fg_save_window_framebuffer(pFullPath, get()));
}
}  // namespace forge

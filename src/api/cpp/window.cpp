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
#include <Window.hpp>

namespace fg
{

Window::Window(int pWidth, int pHeight, const char* pTitle, const Window* pWindow, const bool invisible)
{
    if (pWindow == nullptr) {
        mValue = getHandle(new common::Window(pWidth, pHeight, pTitle, nullptr, invisible));
    } else {
        mValue = getHandle(new common::Window(pWidth, pHeight, pTitle, getWindow(pWindow->get()), invisible));
    }
}

Window::~Window()
{
    delete getWindow(mValue);
}

Window::Window(const Window& other)
{
    mValue = getHandle(new common::Window(other.get()));
}

void Window::setFont(Font* pFont)
{
    getWindow(mValue)->setFont(getFont(pFont->get()));
}

void Window::setTitle(const char* pTitle)
{
    getWindow(mValue)->setTitle(pTitle);
}

void Window::setPos(int pX, int pY)
{
    getWindow(mValue)->setPos(pX, pY);
}

void Window::setSize(unsigned pW, unsigned pH)
{
    getWindow(mValue)->setSize(pW, pH);
}

void Window::setColorMap(ColorMap cmap)
{
    getWindow(mValue)->setColorMap(cmap);
}

long long Window::context() const
{
    return getWindow(mValue)->context();
}

long long Window::display() const
{
    return getWindow(mValue)->display();
}

int Window::width() const
{
    return getWindow(mValue)->width();
}

int Window::height() const
{
    return getWindow(mValue)->height();
}

fg_window Window::get() const
{
    return getWindow(mValue);
}

void Window::hide()
{
    getWindow(mValue)->hide();
}

void Window::show()
{
    getWindow(mValue)->show();
}

bool Window::close()
{
    return getWindow(mValue)->close();
}

void Window::makeCurrent()
{
    getWindow(mValue)->makeCurrent();
}

void Window::draw(const Image& pImage, const bool pKeepAspectRatio)
{
    getWindow(mValue)->draw(getImage(pImage.get()), pKeepAspectRatio);
}

void Window::draw(const Chart& pChart)
{
    getWindow(mValue)->draw(getChart(pChart.get()));
}

void Window::grid(int pRows, int pCols)
{
    getWindow(mValue)->grid(pRows, pCols);
}

void Window::draw(int pColId, int pRowId, const Image& pImage, const char* pTitle, const bool pKeepAspectRatio)
{
    getWindow(mValue)->draw(pColId, pRowId, getImage(pImage.get()), pTitle, pKeepAspectRatio);
}

void Window::draw(int pColId, int pRowId, const Chart& pChart, const char* pTitle)
{
    getWindow(mValue)->draw(pColId, pRowId, getChart(pChart.get()), pTitle);
}

void Window::swapBuffers()
{
    getWindow(mValue)->swapBuffers();
}

void Window::saveFrameBuffer(const char* pFullPath)
{
    getWindow(mValue)->saveFrameBuffer(pFullPath);
}

}

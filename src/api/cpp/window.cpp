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
#include <window.hpp>

namespace forge
{

Window::Window(int pWidth, int pHeight, const char* pTitle, const Window* pWindow, const bool invisible)
{
    try{
        if (pWindow == nullptr) {
            mValue = getHandle(new common::Window(pWidth, pHeight, pTitle, nullptr, invisible));
        } else {
            mValue = getHandle(new common::Window(pWidth, pHeight, pTitle, getWindow(pWindow->get()), invisible));
        }
    } CATCH_INTERNAL_TO_EXTERNAL
}

Window::~Window()
{
    delete getWindow(mValue);
}

Window::Window(const Window& other)
{
    try {
        mValue = getHandle(new common::Window(other.get()));
    } CATCH_INTERNAL_TO_EXTERNAL
}

void Window::setFont(Font* pFont)
{
    try {
        getWindow(mValue)->setFont(getFont(pFont->get()));
    } CATCH_INTERNAL_TO_EXTERNAL
}

void Window::setTitle(const char* pTitle)
{
    try {
        getWindow(mValue)->setTitle(pTitle);
    } CATCH_INTERNAL_TO_EXTERNAL
}

void Window::setPos(int pX, int pY)
{
    try {
        getWindow(mValue)->setPos(pX, pY);
    } CATCH_INTERNAL_TO_EXTERNAL
}

void Window::setSize(unsigned pW, unsigned pH)
{
    try {
        getWindow(mValue)->setSize(pW, pH);
    } CATCH_INTERNAL_TO_EXTERNAL
}

void Window::setColorMap(ColorMap cmap)
{
    try {
        getWindow(mValue)->setColorMap(cmap);
    } CATCH_INTERNAL_TO_EXTERNAL
}

long long Window::context() const
{
    try {
        return getWindow(mValue)->context();
    } CATCH_INTERNAL_TO_EXTERNAL
}

long long Window::display() const
{
    try {
        return getWindow(mValue)->display();
    } CATCH_INTERNAL_TO_EXTERNAL
}

int Window::width() const
{
    try {
        return getWindow(mValue)->width();
    } CATCH_INTERNAL_TO_EXTERNAL
}

int Window::height() const
{
    try {
        return getWindow(mValue)->height();
    } CATCH_INTERNAL_TO_EXTERNAL
}

int Window::gridRows() const
{
    try {
        int rows = 0, cols = 0;
        getWindow(mValue)->getGrid(&rows, &cols);
        return rows;
    } CATCH_INTERNAL_TO_EXTERNAL
}

int Window::gridCols() const
{
    try {
        int rows = 0, cols = 0;
        getWindow(mValue)->getGrid(&rows, &cols);
        return cols;
    } CATCH_INTERNAL_TO_EXTERNAL
}

fg_window Window::get() const
{
    try {
        return getWindow(mValue);
    } CATCH_INTERNAL_TO_EXTERNAL
}

void Window::hide()
{
    try {
        getWindow(mValue)->hide();
    } CATCH_INTERNAL_TO_EXTERNAL
}

void Window::show()
{
    try {
        getWindow(mValue)->show();
    } CATCH_INTERNAL_TO_EXTERNAL
}

bool Window::close()
{
    try {
        return getWindow(mValue)->close();
    } CATCH_INTERNAL_TO_EXTERNAL
}

void Window::makeCurrent()
{
    try {
        getWindow(mValue)->makeCurrent();
    } CATCH_INTERNAL_TO_EXTERNAL
}

void Window::draw(const Image& pImage, const bool pKeepAspectRatio)
{
    try {
        getWindow(mValue)->draw(getImage(pImage.get()), pKeepAspectRatio);
    } CATCH_INTERNAL_TO_EXTERNAL
}

void Window::draw(const Chart& pChart)
{
    try {
        getWindow(mValue)->draw(getChart(pChart.get()));
    } CATCH_INTERNAL_TO_EXTERNAL
}

void Window::grid(int pRows, int pCols)
{
    try {
        getWindow(mValue)->grid(pRows, pCols);
    } CATCH_INTERNAL_TO_EXTERNAL
}

void Window::draw(int pRowId, int pColId, const Image& pImage, const char* pTitle, const bool pKeepAspectRatio)
{
    try {
        getWindow(mValue)->draw(pRowId, pColId, getImage(pImage.get()), pTitle, pKeepAspectRatio);
    } CATCH_INTERNAL_TO_EXTERNAL
}

void Window::draw(int pRowId, int pColId, const Chart& pChart, const char* pTitle)
{
    try {
        getWindow(mValue)->draw(pRowId, pColId, getChart(pChart.get()), pTitle);
    } CATCH_INTERNAL_TO_EXTERNAL
}

void Window::swapBuffers()
{
    try {
        getWindow(mValue)->swapBuffers();
    } CATCH_INTERNAL_TO_EXTERNAL
}

void Window::saveFrameBuffer(const char* pFullPath)
{
    try {
        getWindow(mValue)->saveFrameBuffer(pFullPath);
    } CATCH_INTERNAL_TO_EXTERNAL
}

}

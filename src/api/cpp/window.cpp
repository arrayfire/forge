/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fg/window.h>
#include "error.hpp"

namespace fg
{

Window::Window() : mHandle(0) {}

Window::Window(const uint pWidth, const uint pHeight, const char* pTitle, WindowColorMode pMode, FGenum pChannelType)
        : mHandle(0)
{
    FG_THROW(fg_create_window(&mHandle, pWidth, pHeight, pTitle, pMode, pChannelType));
}

Window::~Window()
{
    FG_THROW(fg_destroy_window(mHandle));
}

uint Window::width() const
{
    if (mHandle==0)
        throw fg::exception("Invalid Window Handle");
    return mHandle->uiWidth;
}

uint Window::height() const
{
    if (mHandle==0)
        throw fg::exception("Invalid Window Handle");
    return mHandle->uiHeight;
}

WindowColorMode Window::colorMode() const
{
    if (mHandle==0)
        throw fg::exception("Invalid Window Handle");
    return mHandle->mode;
}

FGenum Window::channelType() const
{
    if (mHandle==0)
        throw fg::exception("Invalid Window Handle");
    return mHandle->type;
}

fg_window_handle Window::get() const
{
    return mHandle;
}

void makeWindowCurrent(const Window& pWindow)
{
    FG_THROW(fg_make_window_current(pWindow.get()));
}

}

/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fg/window.h>
#include <common.hpp>

namespace backend
{

static fg_window_handle current;

GLEWContext* glewGetContext()
{
    return current->pGLEWContext;
}

void MakeContextCurrent(fg_window_handle wh)
{
    CheckGL("Before MakeContextCurrent");
    if (wh != NULL)
    {
        glfwMakeContextCurrent(wh->pWindow);
        current = wh;
    }
    CheckGL("In MakeContextCurrent");
}

GLenum mode_to_glColor(fg_color_mode mode)
{
    GLenum color = GL_RGBA;
    switch(mode) {
        case FG_RED : color = GL_RED;  break;
        case FG_RGB : color = GL_RGB;  break;
        case FG_RGBA: color = GL_RGBA; break;
    }
    return color;
}

}

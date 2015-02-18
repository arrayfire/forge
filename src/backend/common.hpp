/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

// Parts of this code sourced from SnopyDogy
// https://gist.github.com/SnopyDogy/a9a22497a893ec86aa3e

#include <err_common.hpp>

namespace backend
{
    static WindowHandle current;

    // Required to be defined for GLEW MX to work, along with the GLEW_MX define in the perprocessor!
    static GLEWContext* glewGetContext()
    {
        return current->pGLEWContext;
    }

    static void MakeContextCurrent(WindowHandle wh)
    {
        CheckGL("Before MakeContextCurrent");
        if (wh != NULL)
        {
            glfwMakeContextCurrent(wh->pWindow);
            current = wh;
        }
        CheckGL("In MakeContextCurrent");
    }

    static GLenum mode_to_glColor(afgfx_color_mode mode)
    {
        GLenum color;
        switch(mode) {
            case AFGFX_RED : color = GL_RED;  break;
            case AFGFX_RGB : color = GL_RGB;  break;
            case AFGFX_RGBA: color = GL_RGBA; break;
        }
        return color;
    }
}

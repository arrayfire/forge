/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <fw/defines.h>

typedef struct
{
    GLFWwindow*     pWindow;
    GLEWContext*    pGLEWContext;
    int             uiWidth;
    int             uiHeight;
    int             uiID;
    GLenum          type;
    fw_color_mode   mode;
} fw_window;

typedef fw_window* WindowHandle;

#ifdef __cplusplus
namespace fw
{
}
#endif

#ifdef __cplusplus
extern "C" {
#endif
    FWAPI fw_err fw_create_window(WindowHandle *out, const unsigned width, const unsigned height,
                            const char *title, fw_color_mode mode, GLenum type);

    FWAPI fw_err fw_destroy_window(const WindowHandle in);
#ifdef __cplusplus
}
#endif

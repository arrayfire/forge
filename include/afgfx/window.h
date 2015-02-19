/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <afgfx/defines.h>

typedef struct
{
    GLFWwindow*     pWindow;
    GLEWContext*    pGLEWContext;
    int             uiWidth;
    int             uiHeight;
    int             uiID;
    GLenum          type;
    afgfx_color_mode   mode;
} afgfx_window_struct;

typedef afgfx_window_struct* afgfx_window;

#ifdef __cplusplus
namespace afgfx
{
}
#endif

#ifdef __cplusplus
extern "C" {
#endif
    AFGFXAPI afgfx_err afgfx_create_window(afgfx_window *out, const unsigned width, const unsigned height,
                                           const char *title, afgfx_color_mode mode, GLenum type);

    AFGFXAPI afgfx_err afgfx_make_window_current(const afgfx_window in);

    AFGFXAPI afgfx_err afgfx_destroy_window(const afgfx_window in);
#ifdef __cplusplus
}
#endif

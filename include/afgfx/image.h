/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <afgfx/window.h>

typedef struct
{
    afgfx_window window;

    //OpenGL PBO and texture "names"
    unsigned src_width;
    unsigned src_height;
    GLuint gl_PBO;
    GLuint gl_Tex;
    GLuint gl_Shader;
    GLenum gl_Format;
    GLenum gl_Type;
} afgfx_image_struct;

typedef afgfx_image_struct* afgfx_image;

#ifdef __cplusplus
namespace afgfx
{
}
#endif

#ifdef __cplusplus
extern "C" {
#endif
    AFGFXAPI afgfx_err afgfx_setup_image(afgfx_image *out, const afgfx_window window,
                                         const unsigned width, const unsigned height);

    AFGFXAPI afgfx_err afgfx_draw_image(const afgfx_image in);

    AFGFXAPI afgfx_err afgfx_destroy_image(const afgfx_image in);
#ifdef __cplusplus
}
#endif

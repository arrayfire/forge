/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <fg/window.h>

typedef struct
{
    fg_window_handle window;

    //OpenGL PBO and texture "names"
    unsigned src_width;
    unsigned src_height;
    GLuint gl_PBO;
    GLuint gl_Tex;
    GLuint gl_Shader;
    GLenum gl_Format;
    GLenum gl_Type;
} fg_image_struct;

typedef fg_image_struct* fg_image_handle;

#ifdef __cplusplus
namespace fg
{
}
#endif

#ifdef __cplusplus
extern "C" {
#endif
    FGAPI fg_err fg_setup_image(fg_image_handle *out, const fg_window_handle window,
                                const unsigned width, const unsigned height);

    FGAPI fg_err fg_draw_image(const fg_image_handle in);

    FGAPI fg_err fg_destroy_image(const fg_image_handle in);
#ifdef __cplusplus
}
#endif

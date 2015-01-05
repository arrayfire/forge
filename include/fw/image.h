/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <fw/window.h>

typedef struct
{
    WindowHandle window;

    //OpenGL PBO and texture "names"
    unsigned src_width;
    unsigned src_height;
    GLuint gl_PBO;
    GLuint gl_Tex;
    GLuint gl_Shader;
    GLenum gl_Format;
    GLenum gl_Type;
} fw_image;

typedef fw_image* ImageHandle;

#ifdef __cplusplus
namespace fw
{
}
#endif

#ifdef __cplusplus
extern "C" {
#endif
    FWAPI fw_err fw_setup_image(ImageHandle *out, const WindowHandle window,
                                const unsigned height, const unsigned width);

    FWAPI fw_err fw_draw_image(const ImageHandle in);

    FWAPI fw_err fw_destroy_image(const ImageHandle in);
#ifdef __cplusplus
}
#endif

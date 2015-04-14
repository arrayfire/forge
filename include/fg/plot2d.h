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
/*
typedef struct
{
	GLfloat x;
	GLfloat y;
} point;
*/
typedef struct
{
    fg_window_handle window;
    GLuint gl_vbo;
    size_t vbosize;            // In bytes

    unsigned src_width;
    unsigned src_height;
    GLuint gl_Program;
    GLint gl_Attribute_Coord2d;
    GLint gl_Uniform_Offset_x;
    GLint gl_Uniform_Scale_x;
    GLint gl_Uniform_Offset_y;
    GLint gl_Uniform_Scale_y;
    // TODO: Implement ticks and margins
    int ticksize;
    int margin;

} fg_plot_struct;

typedef fg_plot_struct* fg_plot_handle;

#ifdef __cplusplus
namespace fg
{

class FGAPI Plot2d {
    private:
        fg_plot_handle mHandle;

    public:
        Plot2d();
        Plot2d(fg_plot_handle mHandle, const Window& pWindow, const uint pWidth, const uint pHeight);
        ~Plot2d();

        fg_plot_handle get() const;
        uint width()  const;
        uint height() const;
        FGuint programResourceId() const;
        size_t vbosize() const;
        FGint coord2d()  const;
        FGint offset_x() const;
        FGint scale_x()  const;
        FGint offset_y() const;
        FGint scale_y()  const;
        FGint ticksize() const;
        FGint margin()   const;
};

}
#endif

#ifdef __cplusplus
extern "C" {
#endif
    FGAPI fg_err fg_plot_init(fg_plot_handle *in, const fg_window_handle window, const unsigned width, const unsigned height);

    FGAPI fg_err fg_plot2d(fg_plot_handle in, const double xmax, const double xmin, const double ymax, const double ymin);

    FGAPI fg_err fg_destroy_plot(fg_plot_handle plot);

#ifdef __cplusplus
}
#endif

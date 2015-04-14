/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fg/plot2d.h>
#include "error.hpp"

namespace fg
{

Plot2d::Plot2d() :mHandle(0) {}

Plot2d::Plot2d(fg_plot_handle mHandle, const Window &pWindow, const uint pWidth, const uint pHeight)
{
    FG_THROW(fg_plot_init(&mHandle, pWindow.get(), pWidth, pHeight));
}

Plot2d::~Plot2d()
{
    FG_THROW(fg_destroy_plot(mHandle));
}

uint Plot2d::width() const
{
    if (mHandle==0)
        throw fg::exception("Invalid Plot Handle");
    return mHandle->src_width;
}

uint Plot2d::height() const
{
    if (mHandle==0)
        throw fg::exception("Invalid Plot Handle");
    return mHandle->src_height;
}

FGuint Plot2d::programResourceId() const
{
    if (mHandle==0)
        throw fg::exception("Invalid Plot Handle");
    return mHandle->gl_Program;
}

size_t Plot2d::vbosize() const
{
    if (mHandle==0)
        throw fg::exception("Invalid Plot Handle");
    return mHandle->vbosize;
}

FGint Plot2d::coord2d() const
{
    if (mHandle==0)
        throw fg::exception("Invalid Plot Handle");
    return mHandle->gl_Attribute_Coord2d;
}

FGint Plot2d::offset_x() const
{
    if (mHandle==0)
        throw fg::exception("Invalid Plot Handle");
    return mHandle->gl_Uniform_Offset_x;
}

FGint Plot2d::scale_x() const
{
    if (mHandle==0)
        throw fg::exception("Invalid Plot Handle");
    return mHandle->gl_Uniform_Scale_x;
}

FGint Plot2d::offset_y() const
{
    if (mHandle==0)
        throw fg::exception("Invalid Plot Handle");
    return mHandle->gl_Uniform_Offset_y;
}

FGint Plot2d::scale_y() const
{
    if (mHandle==0)
        throw fg::exception("Invalid Plot Handle");
    return mHandle->gl_Uniform_Scale_y;
}

fg_plot_handle Plot2d::get() const
{
    return mHandle;
}

void drawPlot(const Plot2d& pPlot2d,const double xmax, const double xmin, const double ymax, const double ymin)
{
    FG_THROW(fg_plot2d(pPlot2d.get(), xmax, xmin, ymax, ymin));
}

}

/*******************************************************
* Copyright (c) 2015-2019, ArrayFire
* All rights reserved.
*
* This file is distributed under 3-clause BSD license.
* The complete license agreement can be obtained at:
* http://arrayfire.com/licenses/BSD-3-Clause
********************************************************/

#pragma once

#include <common.hpp>
#include <memory>

namespace forge
{
namespace opengl
{

class colormap_impl {
    private:
        /*
         * READ THIS BEFORE ADDING NEW COLORMAP
         *
         * each of the following buffers will point
         * to the data from floating point arrays
         * defined in cmap.hpp header. Currently,
         * the largest colormap is 506 colors(1520 floats).
         * Hence the shader of internal::image_impl uses
         * uniform array of floats with size 1536 (512 color triplets).
         * when a new colormap is added, make sure
         * the size of array declared in the shaders
         * used by *_impl objects to reflect appropriate
         * size */
        gl::GLuint mDefaultMapBuffer;
        gl::GLuint mSpecMapBuffer;
        gl::GLuint mColorsMapBuffer;
        gl::GLuint mRedMapBuffer;
        gl::GLuint mMoodMapBuffer;
        gl::GLuint mHeatMapBuffer;
        gl::GLuint mBlueMapBuffer;
        /* Current color map lengths */
        gl::GLuint mDefMapLen;
        gl::GLuint mSpecMapLen;
        gl::GLuint mColsMapLen;
        gl::GLuint mRedMapLen;
        gl::GLuint mMoodMapLen;
        gl::GLuint mHeatMapLen;
        gl::GLuint mBlueMapLen;

    public:
        /* constructors and destructors */
        colormap_impl();
        ~colormap_impl();

        gl::GLuint defaultMap() const;
        gl::GLuint spectrum() const;
        gl::GLuint colors() const;
        gl::GLuint red() const;
        gl::GLuint mood() const;
        gl::GLuint heat() const;
        gl::GLuint blue() const;

        gl::GLuint defaultLen() const;
        gl::GLuint spectrumLen() const;
        gl::GLuint colorsLen() const;
        gl::GLuint redLen() const;
        gl::GLuint moodLen() const;
        gl::GLuint heatLen() const;
        gl::GLuint blueLen() const;
};

}
}

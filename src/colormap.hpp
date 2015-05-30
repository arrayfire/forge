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

namespace internal
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
        GLuint mDefaultMapBuffer;
        GLuint mSpecMapBuffer;
        GLuint mColorsMapBuffer;
        GLuint mRedMapBuffer;
        GLuint mMoodMapBuffer;
        GLuint mHeatMapBuffer;
        GLuint mBlueMapBuffer;
        /* Current color map lengths */
        GLuint mDefMapLen;
        GLuint mSpecMapLen;
        GLuint mColsMapLen;
        GLuint mRedMapLen;
        GLuint mMoodMapLen;
        GLuint mHeatMapLen;
        GLuint mBlueMapLen;

    public:
        /* constructors and destructors */
        colormap_impl();
        ~colormap_impl();

        GLuint defaultMap() const;
        GLuint spectrum() const;
        GLuint colors() const;
        GLuint red() const;
        GLuint mood() const;
        GLuint heat() const;
        GLuint blue() const;

        GLuint defaultLen() const;
        GLuint spectrumLen() const;
        GLuint colorsLen() const;
        GLuint redLen() const;
        GLuint moodLen() const;
        GLuint heatLen() const;
        GLuint blueLen() const;
};

}

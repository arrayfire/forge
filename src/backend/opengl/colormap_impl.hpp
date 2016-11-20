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
         * the largest colormap is 259 colors(1036 floats).
         * Hence the shader of internal::image_impl uses
         * uniform array of vec4 with size 259.
         * when a new colormap is added, make sure
         * the size of array declared in the shaders
         * used by *_impl objects to reflect appropriate
         * size */
        gl::GLuint mDefaultMapBuffer;
        gl::GLuint mSpecMapBuffer;
        gl::GLuint mRainbowMapBuffer;
        gl::GLuint mRedMapBuffer;
        gl::GLuint mMoodMapBuffer;
        gl::GLuint mHeatMapBuffer;
        gl::GLuint mBlueMapBuffer;
        gl::GLuint mInfernoMapBuffer;
        gl::GLuint mMagmaMapBuffer;
        gl::GLuint mPlasmaMapBuffer;
        gl::GLuint mViridisMapBuffer;

        /* Current color map lengths */
        gl::GLuint mDefMapLen;
        gl::GLuint mSpecMapLen;
        gl::GLuint mRainbowMapLen;
        gl::GLuint mRedMapLen;
        gl::GLuint mMoodMapLen;
        gl::GLuint mHeatMapLen;
        gl::GLuint mBlueMapLen;
        gl::GLuint mInfernoMapLen;
        gl::GLuint mMagmaMapLen;
        gl::GLuint mPlasmaMapLen;
        gl::GLuint mViridisMapLen;

    public:
        /* constructors and destructors */
        colormap_impl();
        ~colormap_impl();

        gl::GLuint cmapUniformBufferId(forge::ColorMap cmap) const;
        gl::GLuint cmapLength(forge::ColorMap cmap) const;
};

}
}

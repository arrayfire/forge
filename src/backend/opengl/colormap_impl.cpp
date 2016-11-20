/*******************************************************
* Copyright (c) 2015-2019, ArrayFire
* All rights reserved.
*
* This file is distributed under 3-clause BSD license.
* The complete license agreement can be obtained at:
* http://arrayfire.com/licenses/BSD-3-Clause
********************************************************/

#include <common.hpp>
#include <colormap_impl.hpp>
#include <cmap.hpp>

using namespace gl;

#define CREATE_UNIFORM_BUFFER(color_array, size)  \
    createBuffer(GL_UNIFORM_BUFFER, 4*size, color_array, GL_STATIC_DRAW)

namespace forge
{
namespace opengl
{

colormap_impl::colormap_impl()
    :mDefaultMapBuffer(0), mSpecMapBuffer(0), mRainbowMapBuffer(0),
    mRedMapBuffer(0), mMoodMapBuffer(0), mHeatMapBuffer(0),
    mBlueMapBuffer(0)
{
    size_t channel_bytes = sizeof(float)*4; /* 4 is for 4 channels */
    mDefMapLen     = (GLuint)(sizeof(cmap_default)  / channel_bytes);
    mSpecMapLen    = (GLuint)(sizeof(cmap_spectrum) / channel_bytes);
    mRainbowMapLen = (GLuint)(sizeof(cmap_rainbow)   / channel_bytes);
    mRedMapLen     = (GLuint)(sizeof(cmap_red)      / channel_bytes);
    mMoodMapLen    = (GLuint)(sizeof(cmap_mood)     / channel_bytes);
    mHeatMapLen    = (GLuint)(sizeof(cmap_heat)     / channel_bytes);
    mBlueMapLen    = (GLuint)(sizeof(cmap_blue)     / channel_bytes);
    mInfernoMapLen = (GLuint)(sizeof(cmap_inferno)  / channel_bytes);
    mMagmaMapLen   = (GLuint)(sizeof(cmap_magma)    / channel_bytes);
    mPlasmaMapLen  = (GLuint)(sizeof(cmap_plasma)   / channel_bytes);
    mViridisMapLen = (GLuint)(sizeof(cmap_viridis)  / channel_bytes);

    mDefaultMapBuffer = CREATE_UNIFORM_BUFFER(cmap_default, mDefMapLen);
    mSpecMapBuffer    = CREATE_UNIFORM_BUFFER(cmap_spectrum, mSpecMapLen);
    mRainbowMapBuffer = CREATE_UNIFORM_BUFFER(cmap_rainbow, mRainbowMapLen);
    mRedMapBuffer     = CREATE_UNIFORM_BUFFER(cmap_red, mRedMapLen);
    mMoodMapBuffer    = CREATE_UNIFORM_BUFFER(cmap_mood, mMoodMapLen);
    mHeatMapBuffer    = CREATE_UNIFORM_BUFFER(cmap_heat, mHeatMapLen);
    mBlueMapBuffer    = CREATE_UNIFORM_BUFFER(cmap_blue, mBlueMapLen);
    mInfernoMapBuffer = CREATE_UNIFORM_BUFFER(cmap_inferno, mInfernoMapLen);
    mMagmaMapBuffer   = CREATE_UNIFORM_BUFFER(cmap_magma, mMagmaMapLen);
    mPlasmaMapBuffer  = CREATE_UNIFORM_BUFFER(cmap_plasma, mPlasmaMapLen);
    mViridisMapBuffer = CREATE_UNIFORM_BUFFER(cmap_viridis, mViridisMapLen);
}

colormap_impl::~colormap_impl()
{
    glDeleteBuffers(1, &mDefaultMapBuffer);
    glDeleteBuffers(1, &mSpecMapBuffer);
    glDeleteBuffers(1, &mRainbowMapBuffer);
    glDeleteBuffers(1, &mRedMapBuffer);
    glDeleteBuffers(1, &mMoodMapBuffer);
    glDeleteBuffers(1, &mHeatMapBuffer);
    glDeleteBuffers(1, &mBlueMapBuffer);
    glDeleteBuffers(1, &mInfernoMapBuffer);
    glDeleteBuffers(1, &mMagmaMapBuffer);
    glDeleteBuffers(1, &mPlasmaMapBuffer);
    glDeleteBuffers(1, &mViridisMapBuffer);
}

GLuint colormap_impl::cmapUniformBufferId(forge::ColorMap cmap) const
{
    switch(cmap) {
        case FG_COLOR_MAP_SPECTRUM: return mSpecMapBuffer;
        case FG_COLOR_MAP_RAINBOW : return mRainbowMapBuffer;
        case FG_COLOR_MAP_RED     : return mRedMapBuffer;
        case FG_COLOR_MAP_MOOD    : return mMoodMapBuffer;
        case FG_COLOR_MAP_HEAT    : return mHeatMapBuffer;
        case FG_COLOR_MAP_BLUE    : return mBlueMapBuffer;
        case FG_COLOR_MAP_INFERNO : return mInfernoMapBuffer;
        case FG_COLOR_MAP_MAGMA   : return mMagmaMapBuffer;
        case FG_COLOR_MAP_PLASMA  : return mPlasmaMapBuffer;
        case FG_COLOR_MAP_VIRIDIS : return mViridisMapBuffer;
        default: return mDefaultMapBuffer;
    }
}

GLuint colormap_impl::cmapLength(forge::ColorMap cmap) const
{
    switch(cmap) {
        case FG_COLOR_MAP_SPECTRUM: return mSpecMapLen;
        case FG_COLOR_MAP_RAINBOW : return mRainbowMapLen;
        case FG_COLOR_MAP_RED     : return mRedMapLen;
        case FG_COLOR_MAP_MOOD    : return mMoodMapLen;
        case FG_COLOR_MAP_HEAT    : return mHeatMapLen;
        case FG_COLOR_MAP_BLUE    : return mBlueMapLen;
        case FG_COLOR_MAP_INFERNO : return mInfernoMapLen;
        case FG_COLOR_MAP_MAGMA   : return mMagmaMapLen;
        case FG_COLOR_MAP_PLASMA  : return mPlasmaMapLen;
        case FG_COLOR_MAP_VIRIDIS : return mViridisMapLen;
        default: return mDefMapLen;
    }
}

}
}

/*******************************************************
* Copyright (c) 2015-2019, ArrayFire
* All rights reserved.
*
* This file is distributed under 3-clause BSD license.
* The complete license agreement can be obtained at:
* http://arrayfire.com/licenses/BSD-3-Clause
********************************************************/

#include <common.hpp>
#include <colormap.hpp>
#include <cmap.hpp>

#define CREATE_UNIFORM_BUFFER(color_array, size)  \
    createBuffer(GL_UNIFORM_BUFFER, 4*size, color_array, GL_STATIC_DRAW)

namespace internal
{

colormap_impl::colormap_impl()
    :mDefaultMapBuffer(0), mSpecMapBuffer(0), mColorsMapBuffer(0),
    mRedMapBuffer(0), mMoodMapBuffer(0), mHeatMapBuffer(0),
    mBlueMapBuffer(0)
{
    size_t channel_bytes = sizeof(float)*4; /* 4 is for 4 channels */
    mDefMapLen  = (GLuint)(sizeof(cmap_default) /channel_bytes);
    mSpecMapLen = (GLuint)(sizeof(cmap_spectrum)/channel_bytes);
    mColsMapLen = (GLuint)(sizeof(cmap_colors)  /channel_bytes);
    mRedMapLen  = (GLuint)(sizeof(cmap_red)     /channel_bytes);
    mMoodMapLen = (GLuint)(sizeof(cmap_mood)    /channel_bytes);
    mHeatMapLen = (GLuint)(sizeof(cmap_heat)    /channel_bytes);
    mBlueMapLen = (GLuint)(sizeof(cmap_blue)    /channel_bytes);

    mDefaultMapBuffer = CREATE_UNIFORM_BUFFER(cmap_default, mDefMapLen);
    mSpecMapBuffer    = CREATE_UNIFORM_BUFFER(cmap_spectrum, mSpecMapLen);
    mColorsMapBuffer  = CREATE_UNIFORM_BUFFER(cmap_colors, mColsMapLen);
    mRedMapBuffer     = CREATE_UNIFORM_BUFFER(cmap_red, mRedMapLen);
    mMoodMapBuffer    = CREATE_UNIFORM_BUFFER(cmap_mood, mMoodMapLen);
    mHeatMapBuffer    = CREATE_UNIFORM_BUFFER(cmap_heat, mHeatMapLen);
    mBlueMapBuffer    = CREATE_UNIFORM_BUFFER(cmap_blue, mBlueMapLen);
}

colormap_impl::~colormap_impl()
{
    glDeleteBuffers(1, &mDefaultMapBuffer);
    glDeleteBuffers(1, &mSpecMapBuffer);
    glDeleteBuffers(1, &mColorsMapBuffer);
    glDeleteBuffers(1, &mRedMapBuffer);
    glDeleteBuffers(1, &mMoodMapBuffer);
    glDeleteBuffers(1, &mHeatMapBuffer);
    glDeleteBuffers(1, &mBlueMapBuffer);
}

GLuint colormap_impl::defaultMap() const
{
    return mDefaultMapBuffer;
}

GLuint colormap_impl::spectrum() const
{
    return mSpecMapBuffer;
}

GLuint colormap_impl::colors() const
{
    return mColorsMapBuffer;
}

GLuint colormap_impl::red() const
{
    return mRedMapBuffer;
}

GLuint colormap_impl::mood() const
{
    return mMoodMapBuffer;
}

GLuint colormap_impl::heat() const
{
    return mHeatMapBuffer;
}

GLuint colormap_impl::blue() const
{
    return mBlueMapBuffer;
}

GLuint colormap_impl::defaultLen() const
{
    return mDefMapLen;
}

GLuint colormap_impl::spectrumLen() const
{
    return mSpecMapLen;
}

GLuint colormap_impl::colorsLen() const
{
    return mColsMapLen;
}

GLuint colormap_impl::redLen() const
{
    return mRedMapLen;
}

GLuint colormap_impl::moodLen() const
{
    return mMoodMapLen;
}

GLuint colormap_impl::heatLen() const
{
    return mHeatMapLen;
}

GLuint colormap_impl::blueLen() const
{
    return mBlueMapLen;
}

}

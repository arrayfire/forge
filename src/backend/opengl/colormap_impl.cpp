/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <colormap_impl.hpp>
#include <common/cmap.hpp>
#include <gl_helpers.hpp>

#define CREATE_UNIFORM_BUFFER(color_array, size) \
    createBuffer(GL_UNIFORM_BUFFER, 4 * size, color_array, GL_STATIC_DRAW)

namespace forge {
namespace opengl {

colormap_impl::colormap_impl() {
    using namespace forge::common;

    size_t channel_bytes = sizeof(float) * 4; /* 4 is for 4 channels */
    mMapLens[0]          = (uint32_t)(sizeof(cmap_default) / channel_bytes);
    mMapLens[1]          = (uint32_t)(sizeof(cmap_spectrum) / channel_bytes);
    mMapLens[2]          = (uint32_t)(sizeof(cmap_rainbow) / channel_bytes);
    mMapLens[3]          = (uint32_t)(sizeof(cmap_red) / channel_bytes);
    mMapLens[4]          = (uint32_t)(sizeof(cmap_mood) / channel_bytes);
    mMapLens[5]          = (uint32_t)(sizeof(cmap_heat) / channel_bytes);
    mMapLens[6]          = (uint32_t)(sizeof(cmap_blue) / channel_bytes);
    mMapLens[7]          = (uint32_t)(sizeof(cmap_inferno) / channel_bytes);
    mMapLens[8]          = (uint32_t)(sizeof(cmap_magma) / channel_bytes);
    mMapLens[9]          = (uint32_t)(sizeof(cmap_plasma) / channel_bytes);
    mMapLens[10]         = (uint32_t)(sizeof(cmap_viridis) / channel_bytes);

    mMapIds[0]  = CREATE_UNIFORM_BUFFER(cmap_default, mMapLens[0]);
    mMapIds[1]  = CREATE_UNIFORM_BUFFER(cmap_spectrum, mMapLens[0]);
    mMapIds[2]  = CREATE_UNIFORM_BUFFER(cmap_rainbow, mMapLens[0]);
    mMapIds[3]  = CREATE_UNIFORM_BUFFER(cmap_red, mMapLens[0]);
    mMapIds[4]  = CREATE_UNIFORM_BUFFER(cmap_mood, mMapLens[0]);
    mMapIds[5]  = CREATE_UNIFORM_BUFFER(cmap_heat, mMapLens[0]);
    mMapIds[6]  = CREATE_UNIFORM_BUFFER(cmap_blue, mMapLens[0]);
    mMapIds[7]  = CREATE_UNIFORM_BUFFER(cmap_inferno, mMapLens[0]);
    mMapIds[8]  = CREATE_UNIFORM_BUFFER(cmap_magma, mMapLens[0]);
    mMapIds[9]  = CREATE_UNIFORM_BUFFER(cmap_plasma, mMapLens[0]);
    mMapIds[10] = CREATE_UNIFORM_BUFFER(cmap_viridis, mMapLens[0]);
}

colormap_impl::~colormap_impl() {
    glDeleteBuffers(ForgeNumColorMaps, mMapIds.data());
}

uint32_t colormap_impl::cmapUniformBufferId(forge::ColorMap cmap) const {
    return mMapIds[static_cast<unsigned int>(cmap)];
}

uint32_t colormap_impl::cmapLength(forge::ColorMap cmap) const {
    return mMapLens[static_cast<unsigned int>(cmap)];
}

}  // namespace opengl
}  // namespace forge

/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <abstract_renderable.hpp>
#include <fg/defines.h>
#include <shader_program.hpp>

#include <cstdint>
#include <map>

namespace forge {
namespace opengl {

class histogram_impl : public AbstractRenderable {
   private:
    /* plot points characteristics */
    forge::dtype mDataType;
    uint32_t mNBins;
    /* OpenGL Objects */
    ShaderProgram mProgram;
    /* internal shader attributes for mProgram
     * shader program to render histogram bars for each
     * bin*/
    uint32_t mYMaxIndex;
    uint32_t mNBinsIndex;
    uint32_t mMatIndex;
    uint32_t mPointIndex;
    uint32_t mFreqIndex;
    uint32_t mColorIndex;
    uint32_t mAlphaIndex;
    uint32_t mPVCIndex;
    uint32_t mPVAIndex;
    uint32_t mBColorIndex;

    std::map<int, uint32_t> mVAOMap;

    /* bind and unbind helper functions
     * for rendering resources */
    void bindResources(const int pWindowId);
    void unbindResources() const;

   public:
    histogram_impl(const uint32_t pNBins, const forge::dtype pDataType);
    ~histogram_impl();

    void render(const int pWindowId, const int pX, const int pY, const int pVPW,
                const int pVPH, const glm::mat4 &pView,
                const glm::mat4 &pOrient);

    bool isRotatable() const;
};

}  // namespace opengl
}  // namespace forge

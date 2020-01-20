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

#include <glm/glm.hpp>

#include <cstdint>
#include <map>

namespace forge {
namespace opengl {

class pie_impl : public AbstractRenderable {
   protected:
    /* plot points characteristics */
    forge::dtype mDataType;
    uint32_t mNSectors;
    /* OpenGL Objects */
    ShaderProgram mSectorProgram;
    /* shader variable index locations */
    /* vertex shader */
    uint32_t mSectorRangeIndex;
    uint32_t mSectorColorIndex;
    uint32_t mSectorAlphaIndex;
    uint32_t mMaxValueIndex;
    /* geometry shader */
    uint32_t mSectorPVMatIndex;
    /* fragment shader */
    uint32_t mSectorPVCOnIndex;
    uint32_t mSectorPVAOnIndex;

    std::map<int, uint32_t> mVAOMap;

    /* bind and unbind helper functions
     * for rendering resources */
    void bindResources(const int pWindowId);
    void unbindResources() const;

   public:
    pie_impl(const uint32_t pNSectors, const forge::dtype pDataType);
    ~pie_impl();

    virtual void render(const int pWindowId, const int pX, const int pY,
                        const int pVPW, const int pVPH, const glm::mat4 &pView,
                        const glm::mat4 &pOrient);
};

}  // namespace opengl
}  // namespace forge

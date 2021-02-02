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

class vector_field_impl : public AbstractRenderable {
   protected:
    uint32_t mDimension;
    /* plot points characteristics */
    uint32_t mNumPoints;
    forge::dtype mDataType;
    /* OpenGL Objects */
    ShaderProgram mFieldProgram;
    uint32_t mDBO;
    size_t mDBOSize;
    /* shader variable index locations */
    /* vertex shader */
    uint32_t mFieldPointIndex;
    uint32_t mFieldColorIndex;
    uint32_t mFieldAlphaIndex;
    uint32_t mFieldDirectionIndex;
    /* geometry shader */
    uint32_t mFieldPVMatIndex;
    uint32_t mFieldModelMatIndex;
    uint32_t mFieldAScaleMatIndex;
    /* fragment shader */
    uint32_t mFieldPVCOnIndex;
    uint32_t mFieldPVAOnIndex;
    uint32_t mFieldUColorIndex;

    std::map<int, uint32_t> mVAOMap;

    /* bind and unbind helper functions
     * for rendering resources */
    void bindResources(const int pWindowId);
    void unbindResources() const;

    virtual glm::mat4 computeModelMatrix(const glm::mat4& pOrient);

   public:
    vector_field_impl(const uint32_t pNumPoints, const forge::dtype pDataType,
                      const int pDimension = 3);
    ~vector_field_impl();

    uint32_t directions();
    size_t directionsSize() const;

    virtual void render(const int pWindowId, const int pX, const int pY,
                        const int pVPW, const int pVPH, const glm::mat4& pView,
                        const glm::mat4& pOrient);

    virtual bool isRotatable() const;
};

class vector_field2d_impl : public vector_field_impl {
   protected:
    glm::mat4 computeModelMatrix(const glm::mat4& pOrient) override;

   public:
    vector_field2d_impl(const uint32_t pNumPoints, const forge::dtype pDataType)
        : vector_field_impl(pNumPoints, pDataType, 2) {}

    bool isRotatable() const { return false; }
};

}  // namespace opengl
}  // namespace forge

/*******************************************************
* Copyright (c) 2015-2019, ArrayFire
* All rights reserved.
*
* This file is distributed under 3-clause BSD license.
* The complete license agreement can be obtained at:
* http://arrayfire.com/licenses/BSD-3-Clause
********************************************************/

#pragma once

#include <fg/defines.h>
#include <common.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <memory>
#include <map>

namespace opengl
{

class vector_field_impl : public AbstractRenderable {
    protected:
        gl::GLuint    mDimension;
        /* plot points characteristics */
        gl::GLuint    mNumPoints;
        fg::dtype mDataType;
        gl::GLenum    mGLType;
        /* OpenGL Objects */
        ShaderProgram mFieldProgram;
        gl::GLuint    mDBO;
        size_t    mDBOSize;
        /* shader variable index locations */
        /* vertex shader */
        gl::GLuint    mFieldPointIndex;
        gl::GLuint    mFieldColorIndex;
        gl::GLuint    mFieldAlphaIndex;
        gl::GLuint    mFieldDirectionIndex;
        /* geometry shader */
        gl::GLuint    mFieldPVMatIndex;
        gl::GLuint    mFieldModelMatIndex;
        gl::GLuint    mFieldAScaleMatIndex;
        /* fragment shader */
        gl::GLuint    mFieldPVCOnIndex;
        gl::GLuint    mFieldPVAOnIndex;
        gl::GLuint    mFieldUColorIndex;

        std::map<int, gl::GLuint> mVAOMap;

        /* bind and unbind helper functions
         * for rendering resources */
        void bindResources(const int pWindowId);
        void unbindResources() const;

        virtual glm::mat4 computeModelMatrix(const glm::mat4& pOrient);

    public:
        vector_field_impl(const uint pNumPoints, const fg::dtype pDataType,
                          const int pDimension=3);
        ~vector_field_impl();

        gl::GLuint directions();
        size_t directionsSize() const;

        virtual void render(const int pWindowId,
                            const int pX, const int pY, const int pVPW, const int pVPH,
                            const glm::mat4 &pView, const glm::mat4 &pOrient);
};

class vector_field2d_impl : public vector_field_impl {
    protected:
        glm::mat4 computeModelMatrix(const glm::mat4& pOrient) override;
    public:
        vector_field2d_impl(const uint pNumPoints, const fg::dtype pDataType)
            : vector_field_impl(pNumPoints, pDataType, 2) {}
};

}

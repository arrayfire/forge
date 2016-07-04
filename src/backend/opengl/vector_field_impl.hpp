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
        GLuint    mDimension;
        /* plot points characteristics */
        GLuint    mNumPoints;
        fg::dtype mDataType;
        GLenum    mGLType;
        /* OpenGL Objects */
        GLuint    mFieldProgram;
        GLuint    mDBO;
        size_t    mDBOSize;
        /* shader variable index locations */
        /* vertex shader */
        GLuint    mFieldPointIndex;
        GLuint    mFieldColorIndex;
        GLuint    mFieldAlphaIndex;
        GLuint    mFieldDirectionIndex;
        /* geometry shader */
        GLuint    mFieldPVMatIndex;
        GLuint    mFieldModelMatIndex;
        GLuint    mFieldAScaleMatIndex;
        /* fragment shader */
        GLuint    mFieldPVCOnIndex;
        GLuint    mFieldPVAOnIndex;
        GLuint    mFieldUColorIndex;

        std::map<int, GLuint> mVAOMap;

        /* bind and unbind helper functions
         * for rendering resources */
        void bindResources(const int pWindowId);
        void unbindResources() const;

        virtual glm::mat4 computeModelMatrix();

    public:
        vector_field_impl(const uint pNumPoints, const fg::dtype pDataType,
                          const int pDimension=3);
        ~vector_field_impl();

        GLuint directions();
        size_t directionsSize() const;

        virtual void render(const int pWindowId,
                            const int pX, const int pY, const int pVPW, const int pVPH,
                            const glm::mat4 &pView);
};

class vector_field2d_impl : public vector_field_impl {
    protected:
        glm::mat4 computeModelMatrix() override;
    public:
        vector_field2d_impl(const uint pNumPoints, const fg::dtype pDataType)
            : vector_field_impl(pNumPoints, pDataType, 2) {}
};

}

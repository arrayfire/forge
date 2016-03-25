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
#include <map>

namespace opengl
{

class hist_impl : public AbstractRenderable {
    private:
        /* plot points characteristics */
        fg::dtype mDataType;
        GLenum    mGLType;
        GLuint    mNBins;
        /* OpenGL Objects */
        GLuint    mProgram;
        /* internal shader attributes for mProgram
        * shader program to render histogram bars for each
        * bin*/
        GLuint    mYMaxIndex;
        GLuint    mNBinsIndex;
        GLuint    mMatIndex;
        GLuint    mPointIndex;
        GLuint    mFreqIndex;
        GLuint    mColorIndex;
        GLuint    mAlphaIndex;
        GLuint    mPVCIndex;
        GLuint    mPVAIndex;
        GLuint    mBColorIndex;

        std::map<int, GLuint> mVAOMap;

        /* bind and unbind helper functions
         * for rendering resources */
        void bindResources(const int pWindowId);
        void unbindResources() const;
        void computeTransformMat(glm::mat4& pOut, const glm::mat4 pInput);

    public:
        hist_impl(const uint pNBins, const fg::dtype pDataType);
        ~hist_impl();

        void render(const int pWindowId,
                    const int pX, const int pY, const int pVPW, const int pVPH,
                    const glm::mat4& pTransform);
};

}

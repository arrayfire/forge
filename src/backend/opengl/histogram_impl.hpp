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

class histogram_impl : public AbstractRenderable {
    private:
        /* plot points characteristics */
        fg::dtype mDataType;
        gl::GLenum    mGLType;
        gl::GLuint    mNBins;
        /* OpenGL Objects */
        gl::GLuint    mProgram;
        /* internal shader attributes for mProgram
        * shader program to render histogram bars for each
        * bin*/
        gl::GLuint    mYMaxIndex;
        gl::GLuint    mNBinsIndex;
        gl::GLuint    mMatIndex;
        gl::GLuint    mPointIndex;
        gl::GLuint    mFreqIndex;
        gl::GLuint    mColorIndex;
        gl::GLuint    mAlphaIndex;
        gl::GLuint    mPVCIndex;
        gl::GLuint    mPVAIndex;
        gl::GLuint    mBColorIndex;

        std::map<int, gl::GLuint> mVAOMap;

        /* bind and unbind helper functions
         * for rendering resources */
        void bindResources(const int pWindowId);
        void unbindResources() const;

    public:
        histogram_impl(const uint pNBins, const fg::dtype pDataType);
        ~histogram_impl();

        void render(const int pWindowId,
                    const int pX, const int pY, const int pVPW, const int pVPH,
                    const glm::mat4 &pView, const glm::mat4 &pOrient);
};

}

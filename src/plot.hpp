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
#include <chart.hpp>

namespace internal
{

class _Plot : public _Chart {
    private:
        /* plot points characteristics */
        GLuint    mNumPoints;
        GLenum    mDataType;
        float     mLineColor[4];
        /* OpenGL Objects */
        GLuint    mMainVAO;
        GLuint    mMainVBO;
        size_t    mMainVBOsize;

    public:
        _Plot(GLuint pNumPoints, GLenum pDataType);
        ~_Plot();

        void setColor(float r, float g, float b);

        GLuint vbo() const;
        size_t size() const;
        void render(int pX, int pY, int pViewPortWidth, int pViewPortHeight) const;
};

}
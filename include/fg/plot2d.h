/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <fg/chart.h>

namespace fg
{

class FGAPI Plot : public Chart {
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
        Plot(GLuint pNumPoints, GLenum pDataType);
        ~Plot();

        void setColor(float r, float g, float b);

        GLuint vbo() const;
        size_t size() const;
        void render(int pX, int pY, int pViewPortWidth, int pViewPortHeight) const;
};

}

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

class FGAPI Histogram : public Chart {
    private:
        /* plot points characteristics */
        GLenum    mDataType;
        GLuint    mNBins;
        float     mBarColor[4];
        /* OpenGL Objects */
        GLuint    mHistogramVAO;
        GLuint    mHistogramVBO;
        size_t    mHistogramVBOSize;
        GLuint    mHistBarProgram;
        /* internal shader attributes for mHistBarProgram
         * shader program to render histogram bars for each
         * bin*/
        GLuint    mHistBarMatIndex;
        GLuint    mHistBarColorIndex;
        GLuint    mHistBarNBinsIndex;
        GLuint    mHistBarYMaxIndex;

    public:
        Histogram(GLuint pNBins, GLenum pDataType);
        ~Histogram();

        void setBarColor(float r, float g, float b);

        GLuint vbo() const;
        size_t size() const;
        void render(int pX, int pY, int pViewPortWidth, int pViewPortHeight) const;
};

}

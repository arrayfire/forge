/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <fg/window.h>

namespace fg
{

class FGAPI Histogram {
    private:
        GLenum    mDataType;
        GLuint    mVBO[3];
        size_t    mVBOsize;
        GLuint    mProgram;
        GLuint    mAttrCoord2d;
        GLuint    mUnfmColor;
        GLuint    mUnfmTrans;
        int       mTickSize;
        int       mMargin;

        double    mXMax;
        double    mXMin;
        double    mYMax;
        double    mYMin;

    public:
        Histogram(GLenum pDataType);
        ~Histogram();

        GLuint vbo() const;
        size_t size() const;

        void setAxesLimits(double pXmax, double pXmin, double pYmax, double pYmin);
        void setVBOSize(size_t pSize);

        void render(int pViewPortWidth, int pViewPortHeight, int size) const;
};

FGAPI void drawHistogram(Window* pWindow, const Histogram& pHistogram, const unsigned int nbins, const double minval, const double maxval);

}

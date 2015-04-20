/*******************************************************
 * Copyright (c) 2014, ArrayFire
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

class FGAPI Plot {
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
        Plot(GLenum pDataType);
        ~Plot();

        GLuint vbo() const;
        size_t size() const;
        double xmax() const;
        double xmin() const;
        double ymax() const;
        double ymin() const;

        void setAxesLimits(double pXmax, double pXmin, double pYmax, double pYmin);
        void setVBOSize(size_t pSize);

        void render(int pViewPortWidth, int pViewPortHeight) const;
};

FGAPI void drawPlot(Window* pWindow, const Plot& pPlot);

}

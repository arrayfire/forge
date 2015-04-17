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
        size_t    mVBOSize;
        GLuint    mProgram;
        GLuint    mAttrCoord2d;
        GLuint    mUnfmColor;
        GLuint    mUnfmTrans;
        int       mTickSize;
        int       mMargin;

    public:
        Plot(GLenum pDataType);
        ~Plot();

        GLuint vbo() const;
        size_t vboSize() const;
        void setVBOSize(size_t pSize);

        void render(int pViewPortWidth, int pViewPortHeight,
                    double pXmax, double pXmin,
                    double pYmax, double pYmin) const;
};

FGAPI void drawPlot(Window* pWindow, const Plot& pPlot,
              const double pXmax=0, const double pXmin=0,
              const double pYmax=0, const double pYmin=0);

}

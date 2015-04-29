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

namespace fg
{

class FGAPI Plot {
    private:
        /* internal class attributes for
         * drawing ticks on axes for plots*/
        int       mTickCount;
        int       mTickSize;
        int       mMargin;
        /* plot points characteristics */
        double    mXMax;
        double    mXMin;
        double    mYMax;
        double    mYMin;
        float     mLineColor[4];
        GLuint    mNumPoints;
        GLenum    mDataType;
        /* OpenGL Objects */
        GLuint    mMainVAO;
        GLuint    mMainVBO;
        GLuint    mDecorVAO;
        GLuint    mDecorVBO;
        size_t    mMainVBOsize;
        GLuint    mProgram;
        GLuint    mSpriteProgram;
        /* shader uniform variable locations */
        GLuint    mAttrCoord2d;
        GLuint    mUnfmColor;
        GLuint    mUnfmTrans;
        GLuint    mUnfmTickTrans;
        GLuint    mUnfmTickColor;
        GLuint    mUnfmTickAxis;
        /* private helper functions */
        void createDecorVAO();

    public:
        Plot(GLuint pNumPoints, GLenum pDataType);
        ~Plot();

        GLuint vbo() const;
        size_t size() const;
        double xmax() const;
        double xmin() const;
        double ymax() const;
        double ymin() const;

        void setAxesLimits(double pXmax, double pXmin, double pYmax, double pYmin);
        void setColor(float r, float g, float b);

        void render(int pViewPortWidth, int pViewPortHeight) const;
};

}

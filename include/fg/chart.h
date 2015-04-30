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

class FGAPI Chart {
    private:
        /* internal class attributes for
         * drawing ticks on axes for plots*/
        int       mTickCount;
        int       mTickSize;
        int       mMargin;
        /* chart characteristics */
        double    mXMax;
        double    mXMin;
        double    mYMax;
        double    mYMin;
        /* OpenGL Objects */
        GLuint    mDecorVAO;
        GLuint    mDecorVBO;
        GLuint    mBorderProgram;
        GLuint    mSpriteProgram;
        /* shader uniform variable locations */
        GLuint    mBorderPointIndex;
        GLuint    mBorderColorIndex;
        GLuint    mBorderMatIndex;
        GLuint    mSpriteMatIndex;
        GLuint    mSpriteTickcolorIndex;
        GLuint    mSpriteTickaxisIndex;

    protected:
        void bindBorderProgram() const;
        void unbindBorderProgram() const;
        GLuint rectangleVBO() const;
        GLuint borderProgramPointIndex() const;
        GLuint borderColorIndex() const;
        GLuint borderMatIndex() const;
        int tickSize() const;
        int margin() const;

    public:
        Chart();
        virtual ~Chart();

        double xmax() const;
        double xmin() const;
        double ymax() const;
        double ymin() const;
        void setAxesLimits(double pXmax, double pXmin, double pYmax, double pYmin);
        void renderChart(int pViewPortWidth, int pViewPortHeight) const;

        virtual GLuint vbo() const = 0;
        virtual size_t size() const = 0;
        virtual void render(int pViewPortWidth, int pViewPortHeight) const = 0;
};

}

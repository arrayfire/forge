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
#include <vector>
#include <string>
#include <map>

namespace internal
{

class AbstractChart2D : public AbstractRenderable {
    private:
        /* internal class attributes for
        * drawing ticks on axes for plots*/
        std::vector<float> mTickTextX;
        std::vector<float> mTickTextY;
        std::vector<std::string> mXText;
        std::vector<std::string> mYText;
        int       mTickCount;
        int       mTickSize;
        int       mLeftMargin;
        int       mRightMargin;
        int       mTopMargin;
        int       mBottomMargin;
        /* chart characteristics */
        float    mXMax;
        float    mXMin;
        float    mYMax;
        float    mYMin;
        std::string mXTitle;
        std::string mYTitle;
        /* OpenGL Objects */
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
        /* VAO map to store a vertex array object
         * for each valid window context */
        std::map<const void*, GLuint> mVAOMap;

        void bindResources(const void* pWnd);
        void unbindResources() const;

    protected:
        void bindBorderProgram() const;
        void unbindBorderProgram() const;
        GLuint rectangleVBO() const;
        GLuint borderProgramPointIndex() const;
        GLuint borderColorIndex() const;
        GLuint borderMatIndex() const;
        int tickSize() const;
        int leftMargin() const;
        int rightMargin() const;
        int bottomMargin() const;
        int topMargin() const;

        /* this function should be used after
        * setAxesLimits is called as this function uses
        * those values to generate the text markers that
        * are placed near the ticks on the axes */
        void setTickCount(int pTickCount);

    public:
        AbstractChart2D();
        virtual ~AbstractChart2D();

        void setAxesLimits(float pXmax, float pXmin, float pYmax, float pYmin);
        void setXAxisTitle(const char* pTitle);
        void setYAxisTitle(const char* pTitle);

        float xmax() const;
        float xmin() const;
        float ymax() const;
        float ymin() const;
        void renderChart(const void* pWnd, int pX, int pY, int pViewPortWidth, int pViewPortHeight);

        virtual GLuint vbo() const = 0;
        virtual size_t size() const = 0;
        virtual void render(const void* pWnd, int pX, int pY, int pViewPortWidth, int pViewPortHeight) = 0;
};

}

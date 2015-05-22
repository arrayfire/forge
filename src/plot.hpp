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
#include <memory>

namespace internal
{

class plot_impl : public _Chart {
    public:
        /* plot points characteristics */
        GLuint    mNumPoints;
        GLenum    mDataType;
        float     mLineColor[4];
        /* OpenGL Objects */
        GLuint    mMainVAO;
        GLuint    mMainVBO;
        size_t    mMainVBOsize;

        plot_impl(GLuint pNumPoints, GLenum pDataType);
        ~plot_impl();

        inline GLuint vbo() const { return mMainVBO; }
        inline size_t size() const { return mMainVBOsize; }

        void render(int pX, int pY, int pViewPortWidth, int pViewPortHeight) const;
};

class _Plot {
    private:
        std::shared_ptr<plot_impl> plt;

    public:
        _Plot(GLuint pNumPoints, GLenum pDataType)
            : plt(std::make_shared<plot_impl>(pNumPoints, pDataType)) {}

        void setColor(float r, float g, float b) {
            plt->mLineColor[0] = clampTo01(r);
            plt->mLineColor[1] = clampTo01(g);
            plt->mLineColor[2] = clampTo01(b);
            plt->mLineColor[3] = 1.0f;
        }

        inline void setAxesLimits(float pXmax, float pXmin, float pYmax, float pYmin) {
            plt->setAxesLimits(pXmax, pXmin, pYmax, pYmin);
        }

        inline void setXAxisTitle(const char* pTitle) { plt->setXAxisTitle(pTitle); }
        inline void setYAxisTitle(const char* pTitle) { plt->setYAxisTitle(pTitle); }

        inline float xmax() const { return plt->xmax(); }
        inline float xmin() const { return plt->xmin(); }
        inline float ymax() const { return plt->ymax(); }
        inline float ymin() const { return plt->ymin(); }
        inline GLuint vbo() const { return plt->vbo();  }
        inline size_t size() const { return plt->size();}

        inline void render(int pX, int pY, int pViewPortWidth, int pViewPortHeight) const {
            plt->render(pX, pY, pViewPortWidth, pViewPortHeight);
        }
};

}
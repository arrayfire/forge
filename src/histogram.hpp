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

class hist_impl : public _Chart {
    public:
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

        hist_impl(GLuint pNBins, GLenum pDataType);
        ~hist_impl();

        inline GLuint vbo() const { return mHistogramVBO; }
        inline size_t size() const { return mHistogramVBOSize; }

        void render(int pX, int pY, int pViewPortWidth, int pViewPortHeight) const;
};

class _Histogram : public _Chart {
    private:
        std::shared_ptr<hist_impl> hst;

    public:
        _Histogram(GLuint pNBins, GLenum pDataType)
            : hst(std::make_shared<hist_impl>(pNBins, pDataType)) {}

        void setBarColor(float r, float g, float b) {
            hst->mBarColor[0] = r;
            hst->mBarColor[1] = g;
            hst->mBarColor[2] = b;
            hst->mBarColor[3] = 1.0f;
        }

        inline void setAxesLimits(float pXmax, float pXmin, float pYmax, float pYmin) {
            hst->setAxesLimits(pXmax, pXmin, pYmax, pYmin);
        }

        inline void setXAxisTitle(const char* pTitle) { hst->setXAxisTitle(pTitle); }
        inline void setYAxisTitle(const char* pTitle) { hst->setYAxisTitle(pTitle); }

        inline float xmax() const { return hst->xmax(); }
        inline float xmin() const { return hst->xmin(); }
        inline float ymax() const { return hst->ymax(); }
        inline float ymin() const { return hst->ymin(); }
        inline GLuint vbo() const { return hst->vbo(); }
        inline size_t size() const { return hst->size(); }

        inline void render(int pX, int pY, int pViewPortWidth, int pViewPortHeight) const {
            hst->render(pX, pY, pViewPortWidth, pViewPortHeight);
        }
};

}
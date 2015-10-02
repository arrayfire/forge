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
#include <map>

namespace internal
{

class hist_impl : public AbstractChart2D {
    private:
        /* plot points characteristics */
        fg::dtype mDataType;
        GLenum    mGLType;
        GLuint    mNBins;
        float     mBarColor[4];
        /* OpenGL Objects */
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
        GLuint    mPointIndex;
        GLuint    mFreqIndex;

        std::map<int, GLuint> mVAOMap;

        /* bind and unbind helper functions
         * for rendering resources */
        void bindResources(int pWindowId);
        void unbindResources() const;

    public:
        hist_impl(unsigned pNBins, fg::dtype pDataType);
        ~hist_impl();

        void setBarColor(float r, float g, float b);
        GLuint vbo() const;
        size_t size() const;

        void render(int pWindowId, int pX, int pY, int pViewPortWidth, int pViewPortHeight);
};

class _Histogram {
    private:
        std::shared_ptr<hist_impl> hst;

    public:
        _Histogram(unsigned pNBins, fg::dtype pDataType)
            : hst(std::make_shared<hist_impl>(pNBins, pDataType)) {}

        inline const std::shared_ptr<hist_impl>& impl() const {
            return hst;
        }

        inline void setBarColor(float r, float g, float b) {
            hst->setBarColor(r, g, b);
        }

        inline void setAxesLimits(float pXmax, float pXmin, float pYmax, float pYmin) {
            hst->setAxesLimits(pXmax, pXmin, pYmax, pYmin);
        }

        inline void setXAxisTitle(const char* pTitle) {
            hst->setXAxisTitle(pTitle);
        }

        inline void setYAxisTitle(const char* pTitle) {
            hst->setYAxisTitle(pTitle);
        }

        inline float xmax() const {
            return hst->xmax();
        }

        inline float xmin() const {
            return hst->xmin();
        }

        inline float ymax() const {
            return hst->ymax();
        }

        inline float ymin() const {
            return hst->ymin();
        }

        inline GLuint vbo() const {
            return hst->vbo();
        }

        inline size_t size() const {
            return hst->size();
        }
};

}

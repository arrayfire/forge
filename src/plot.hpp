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
#include <glm/glm.hpp>

namespace internal
{

class plot_impl : public Chart2D {
    protected:
        /* plot points characteristics */
        GLuint    mNumPoints;
        fg::dtype mDataType;
        GLenum    mGLType;
        float     mLineColor[4];
        fg::MarkerType mMarkerType;
        fg::PlotType   mPlotType;
        /* OpenGL Objects */
        GLuint    mMainVBO;
        size_t    mMainVBOsize;
        GLuint    mMarkerProgram;
        /* shared variable index locations */
        GLuint    mPointIndex;
        GLuint    mMarkerColIndex;
        GLuint    mMarkerTypeIndex;
        GLuint    mSpriteTMatIndex;

        std::map<int, GLuint> mVAOMap;

        /* bind and unbind helper functions
         * for rendering resources */
        void bindResources(int pWindowId);
        void unbindResources() const;

    public:
        plot_impl(unsigned pNumPoints, fg::dtype pDataType, fg::PlotType, fg::MarkerType);
        ~plot_impl();

        void setColor(fg::Color col);
        void setColor(float r, float g, float b);
        GLuint vbo() const;
        size_t size() const;

        void render(int pWindowId, int pX, int pY, int pViewPortWidth, int pViewPortHeight);
};

class _Plot {
    private:
        std::shared_ptr<plot_impl> plt;

    public:
        _Plot(unsigned pNumPoints, fg::dtype pDataType, fg::PlotType pType, fg::MarkerType mType)
            : plt(std::make_shared<plot_impl>(pNumPoints, pDataType, pType, mType)) {}

        inline const std::shared_ptr<plot_impl>& impl() const {
            return plt;
        }

        inline void setColor(fg::Color col) {
            plt->setColor(col);
        }

        inline void setColor(float r, float g, float b) {
            plt->setColor(r, g, b);
        }

        inline void setAxesLimits(float pXmax, float pXmin, float pYmax, float pYmin) {
            plt->setAxesLimits(pXmax, pXmin, pYmax, pYmin);
        }

        inline void setAxesTitles(const char* pXTitle, const char* pYTitle) {
            plt->setAxesTitles(pXTitle, pYTitle);
        }

        inline float xmax() const {
            return plt->xmax();
        }

        inline float xmin() const {
            return plt->xmin();
        }

        inline float ymax() const {
            return plt->ymax();
        }

        inline float ymin() const {
            return plt->ymin();
        }

        inline GLuint vbo() const {
            return plt->vbo();
        }

        inline size_t size() const {
            return plt->size();
        }
};

}

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

class plot3_impl : public Chart3D {
    protected:
        /* plot points characteristics */
        GLuint    mNumPoints;
        GLenum    mDataType;
        float     mLineColor[4];
        fg::MarkerType mMarkerType;
        fg::PlotType mPlotType;
        /* OpenGL Objects */
        GLuint    mMainVBO;
        size_t    mMainVBOsize;
        size_t    mIndexVBOsize;
        GLuint    mMarkerProgram;
        GLuint    mPlot3Program;
        /* shared variable index locations */
        GLuint    mPointIndex;
        GLuint    mMarkerTypeIndex;
        GLuint    mMarkerColIndex;
        GLuint    mSpriteTMatIndex;
        GLuint    mPlot3PointIndex;
        GLuint    mPlot3TMatIndex;
        GLuint    mPlot3RangeIndex;

        std::map<int, GLuint> mVAOMap;

        /* bind and unbind helper functions
         * for rendering resources */
        void bindResources(int pWindowId);
        void unbindResources() const;

    public:
        plot3_impl(unsigned pNumPoints, fg::dtype pDataType, fg::PlotType pPlotType, fg::MarkerType pMarkerType);
        ~plot3_impl();

        void setColor(fg::Color col);
        void setColor(float r, float g, float b);
        GLuint vbo() const;
        size_t size() const;

        void render(int pWindowId, int pX, int pY, int pViewPortWidth, int pViewPortHeight);
};

class _Plot3 {
    private:
        std::shared_ptr<plot3_impl> plt;

    public:
        _Plot3(unsigned pNumPoints, fg::dtype pDataType, fg::PlotType pPlotType=fg::FG_LINE, fg::MarkerType pMarkerType=fg::FG_NONE) {
            plt = std::make_shared<plot3_impl>(pNumPoints, pDataType, pPlotType, pMarkerType);
        }

        inline const std::shared_ptr<plot3_impl>& impl() const {
            return plt;
        }

        inline void setColor(fg::Color col) {
            plt->setColor(col);
        }

        inline void setColor(float r, float g, float b) {
            plt->setColor(r, g, b);
        }

        inline void setAxesLimits(float pXmax, float pXmin, float pYmax, float pYmin, float pZmax, float pZmin) {
            plt->setAxesLimits(pXmax, pXmin, pYmax, pYmin, pZmax, pZmin);
        }

        inline void setAxesTitles(const char* pXTitle, const char* pYTitle, const char* pZTitle)
        {
            plt->setAxesTitles(pXTitle, pYTitle, pZTitle);
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

        inline float zmax() const {
            return plt->zmax();
        }

        inline float zmin() const {
            return plt->zmin();
        }

        inline GLuint vbo() const {
            return plt->vbo();
        }

        inline size_t size() const {
            return plt->size();
        }
};

}

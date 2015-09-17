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
#include <scatter.hpp>
#include <memory>
#include <map>

namespace internal
{

class plot_impl : public AbstractChart2D {
    private:
        /* plot points characteristics */
        GLuint    mNumPoints;
        GLenum    mDataType;
        float     mLineColor[4];
        fg::FGMarkerType mMarkerType;
        /* OpenGL Objects */
        GLuint    mMainVBO;
        size_t    mMainVBOsize;
        GLuint    mMarkerProgram;
        /* shared variable index locations */
        GLuint    mPointIndex;
        GLuint    mMarkerTypeIndex;
        GLuint    mSpriteTMatIndex;

        std::map<int, GLuint> mVAOMap;

        /* bind and unbind helper functions
         * for rendering resources */
        void bindResources(int pWindowId);
        void unbindResources() const;
        GLuint markerTypeIndex() const;
        GLuint spriteMatIndex() const;

    public:
        plot_impl(unsigned pNumPoints, fg::FGType pDataType, fg::FGMarkerType=fg::FG_NONE);
        ~plot_impl();

        void setColor(float r, float g, float b);
        GLuint vbo() const;
        size_t size() const;

        void render(int pWindowId, int pX, int pY, int pViewPortWidth, int pViewPortHeight);
};

class _Plot {
    private:
        std::shared_ptr<plot_impl> plt;

    public:
        _Plot(unsigned pNumPoints, fg::FGType pDataType, fg::FGPlotType pPlotType=fg::FG_LINE, fg::FGMarkerType pMarkerType=fg::FG_NONE){
            switch(pPlotType){
                case(fg::FG_LINE):
                    plt = std::make_shared<plot_impl>(pNumPoints, pDataType, pMarkerType);
                    break;
                case(fg::FG_SCATTER):
                    plt = std::make_shared<scatter_impl>(pNumPoints, pDataType, pMarkerType);
                    break;
                default:
                    plt = std::make_shared<plot_impl>(pNumPoints, pDataType, pMarkerType);
            };
        }

        inline const std::shared_ptr<plot_impl>& impl() const {
            return plt;
        }

        inline void setColor(float r, float g, float b) {
            plt->setColor(r, g, b);
        }

        inline void setAxesLimits(float pXmax, float pXmin, float pYmax, float pYmin) {
            plt->setAxesLimits(pXmax, pXmin, pYmax, pYmin);
        }

        inline void setXAxisTitle(const char* pTitle) {
            plt->setXAxisTitle(pTitle);
        }

        inline void setYAxisTitle(const char* pTitle) {
            plt->setYAxisTitle(pTitle);
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

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

class plot_impl : public AbstractChart2D {
    protected:
        /* plot points characteristics */
        GLuint    mNumPoints;
        fg::dtype mDataType;
        GLenum    mGLType;
        float     mLineColor[4];
        fg::MarkerType mMarkerType;
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
        virtual void renderGraph(int pWindowId, glm::mat4 transform);

    public:
        plot_impl(unsigned pNumPoints, fg::dtype pDataType, fg::MarkerType=fg::FG_NONE);
        ~plot_impl();

        void setColor(fg::Color col);
        void setColor(float r, float g, float b);
        GLuint vbo() const;
        size_t size() const;

        void render(int pWindowId, int pX, int pY, int pViewPortWidth, int pViewPortHeight);
};

class scatter_impl : public plot_impl {
   private:
        void renderGraph(int pWindowId, glm::mat4 transform);

   public:
       scatter_impl(unsigned pNumPoints, fg::dtype pDataType, fg::MarkerType pMarkerType=fg::FG_NONE) : plot_impl(pNumPoints, pDataType, pMarkerType)   {}
       ~scatter_impl() {}
};

class _Plot {
    private:
        std::shared_ptr<plot_impl> plt;

    public:
        _Plot(unsigned pNumPoints, fg::dtype pDataType, fg::PlotType pPlotType=fg::FG_LINE, fg::MarkerType pMarkerType=fg::FG_NONE){
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

        inline void setColor(fg::Color col) {
            plt->setColor(col);
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

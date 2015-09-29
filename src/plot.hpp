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
        virtual void renderGraph(int pWindowId, glm::mat4 transform);

    public:
        plot_impl(unsigned pNumPoints, fg::FGType pDataType, fg::FGMarkerType=fg::FG_NONE);
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
       scatter_impl(unsigned pNumPoints, fg::FGType pDataType, fg::FGMarkerType pMarkerType=fg::FG_NONE) : plot_impl(pNumPoints, pDataType, pMarkerType)   {}
       ~scatter_impl() {}
};

class plot3_impl : public AbstractChart3D {
    protected:
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
        virtual void renderGraph(int pWindowId, glm::mat4 transform);

   public:
       plot3_impl(unsigned pNumPoints, fg::FGType pDataType, fg::FGMarkerType pMarkerType=fg::FG_NONE);
       ~plot3_impl();

        void setColor(fg::Color col);
        void setColor(float r, float g, float b);
        GLuint vbo() const;
        size_t size() const;

        void render(int pWindowId, int pX, int pY, int pViewPortWidth, int pViewPortHeight);
};

class _Plot {
    private:
        std::shared_ptr<plot_impl> plt;
        std::shared_ptr<plot3_impl> plt3;
        fg::FGPlotType mPlotType;

    public:
        _Plot(unsigned pNumPoints, fg::FGType pDataType, fg::FGPlotType pPlotType=fg::FG_LINE, fg::FGMarkerType pMarkerType=fg::FG_NONE){
            switch(pPlotType){
                case(fg::FG_LINE):
                    plt  = std::make_shared<plot_impl>(pNumPoints, pDataType, pMarkerType);
                    plt3 = NULL;
                    break;
                case(fg::FG_SCATTER):
                    plt  = std::make_shared<scatter_impl>(pNumPoints, pDataType, pMarkerType);
                    plt3 = NULL;
                    break;
                case(fg::FG_LINE_3D):
                    plt  = NULL;
                    plt3 = std::make_shared<plot3_impl>(pNumPoints, pDataType, pMarkerType);
                    break;
                default:
                    plt = std::make_shared<plot_impl>(pNumPoints, pDataType, pMarkerType);
            };
            mPlotType = pPlotType;
        }

        inline const std::shared_ptr<plot_impl>& impl() const {
            //Handle backend implementation? overload? inheritance?
            return plt;
        }

        inline void setColor(fg::Color col) {
            if(mPlotType == fg::FG_LINE_3D)
                plt3->setColor(col);
            else
                plt->setColor(col);
        }

        inline void setColor(float r, float g, float b) {
            if(mPlotType == fg::FG_LINE_3D)
                plt3->setColor(r, g, b);
            else
                plt->setColor(r, g, b);
        }

        inline void setAxesLimits(float pXmax, float pXmin, float pYmax, float pYmin) {
                plt->setAxesLimits(pXmax, pXmin, pYmax, pYmin);
        }

        inline void setAxesLimits(float pXmax, float pXmin, float pYmax, float pYmin, float pZmax, float pZmin) {
            if(mPlotType == fg::FG_LINE_3D)
                plt3->setAxesLimits(pXmax, pXmin, pYmax, pYmin, pZmax, pZmin);
        }

        inline void setXAxisTitle(const char* pTitle) {
            if(mPlotType == fg::FG_LINE_3D)
                plt3->setXAxisTitle(pTitle);
            else
                plt->setXAxisTitle(pTitle);
        }

        inline void setYAxisTitle(const char* pTitle) {
            if(mPlotType == fg::FG_LINE_3D)
                plt3->setYAxisTitle(pTitle);
            else
                plt->setYAxisTitle(pTitle);
        }

        inline float xmax() const {
            if(mPlotType == fg::FG_LINE_3D)
                return plt3->xmax();
            else
                return plt->xmax();
        }

        inline float xmin() const {
            if(mPlotType == fg::FG_LINE_3D)
                return plt3->xmin();
            else
                return plt->xmin();
        }

        inline float ymax() const {
            if(mPlotType == fg::FG_LINE_3D)
                return plt3->ymax();
            else
                return plt->ymax();
        }

        inline float ymin() const {
            if(mPlotType == fg::FG_LINE_3D)
                return plt3->ymin();
            else
                return plt->ymin();
        }

        inline GLuint vbo() const {
            if(mPlotType == fg::FG_LINE_3D)
                return plt3->vbo();
            else
                return plt->vbo();
        }

        inline size_t size() const {
            if(mPlotType == fg::FG_LINE_3D)
                return plt3->size();
            else
                return plt->size();
        }
};

}

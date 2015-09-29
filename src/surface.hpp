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

class surface_impl : public AbstractChart3D {
    protected:
        /* plot points characteristics */
        GLuint    mNumXPoints;
        GLuint    mNumYPoints;
        GLenum    mDataType;
        float     mLineColor[4];
        fg::MarkerType mMarkerType;
        /* OpenGL Objects */
        GLuint    mMainVBO;
        size_t    mMainVBOsize;
        GLuint    mIndexVBO;
        size_t    mIndexVBOsize;
        GLuint    mMarkerProgram;
        GLuint    mSurfProgram;
        /* shared variable index locations */
        GLuint    mPointIndex;
        GLuint    mMarkerTypeIndex;
        GLuint    mMarkerColIndex;
        GLuint    mSpriteTMatIndex;
        GLuint    mSurfPointIndex;
        GLuint    mSurfTMatIndex;
        GLuint    mSurfRangeIndex;

        std::map<int, GLuint> mVAOMap;

        /* bind and unbind helper functions
         * for rendering resources */
        void bindResources(int pWindowId);
        void unbindResources() const;
        void bindSurfProgram() const;
        void unbindSurfProgram() const;
        GLuint markerTypeIndex() const;
        GLuint spriteMatIndex() const;
        GLuint markerColIndex() const;
        GLuint surfRangeIndex() const;
        GLuint surfMatIndex() const;
        virtual void renderGraph(int pWindowId, glm::mat4 transform);

    public:
        surface_impl(unsigned pNumXpoints, unsigned pNumYpoints, fg::dtype pDataType, fg::MarkerType pMarkerType);
        ~surface_impl();

        void setColor(fg::Color col);
        void setColor(float r, float g, float b);
        GLuint vbo() const;
        size_t size() const;

        void render(int pWindowId, int pX, int pY, int pViewPortWidth, int pViewPortHeight);
};

class scatter3_impl : public surface_impl {
   private:
        void renderGraph(int pWindowId, glm::mat4 transform);

   public:
       scatter3_impl(unsigned pNumXPoints, unsigned pNumYPoints, fg::dtype pDataType, fg::MarkerType pMarkerType=fg::FG_NONE) : surface_impl(pNumXPoints, pNumYPoints, pDataType, pMarkerType)   {}
       ~scatter3_impl() {}
};

class _Surface {
    private:
        std::shared_ptr<surface_impl> plt;

    public:
        _Surface(unsigned pNumXPoints, unsigned pNumYPoints, fg::dtype pDataType, fg::PlotType pPlotType=fg::FG_SURFACE, fg::MarkerType pMarkerType=fg::FG_NONE){
            switch(pPlotType){
                case(fg::FG_SURFACE):
                    plt = std::make_shared<surface_impl>(pNumXPoints, pNumYPoints, pDataType, pMarkerType);
                    break;
                case(fg::FG_SCATTER):
                    plt = std::make_shared<scatter3_impl>(pNumXPoints, pNumYPoints, pDataType, pMarkerType);
                    break;
                default:
                    plt = std::make_shared<surface_impl>(pNumXPoints, pNumYPoints, pDataType, pMarkerType);
            };
        }

        inline const std::shared_ptr<surface_impl>& impl() const {
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

        inline void setXAxisTitle(const char* pTitle) {
            plt->setXAxisTitle(pTitle);
        }

        inline void setYAxisTitle(const char* pTitle) {
            plt->setYAxisTitle(pTitle);
        }

        inline void setZAxisTitle(const char* pTitle) {
            plt->setZAxisTitle(pTitle);
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

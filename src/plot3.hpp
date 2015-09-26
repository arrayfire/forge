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

class plot3_impl : public AbstractChart3D {
    protected:
        /* plot points characteristics */
        GLuint    mNumXPoints;
        GLuint    mNumYPoints;
        GLenum    mDataType;
        float     mLineColor[4];
        fg::FGMarkerType mMarkerType;
        /* OpenGL Objects */
        GLuint    mMainVBO;
        size_t    mMainVBOsize;
        GLuint    mIndexVBO;
        size_t    mIndexVBOsize;
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
        plot3_impl(unsigned pNumXpoints, unsigned pNumYpoints, fg::FGType pDataType);
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
        _Plot3(unsigned pNumXpoints, unsigned pNumYpoints, fg::FGType pDataType){
                plt = std::make_shared<plot3_impl>(pNumXpoints, pNumYpoints, pDataType);
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

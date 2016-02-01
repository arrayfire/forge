/*******************************************************
* Copyright (c) 2015-2019, ArrayFire
* All rights reserved.
*
* This file is distributed under 3-clause BSD license.
* The complete license agreement can be obtained at:
* http://arrayfire.com/licenses/BSD-3-Clause
********************************************************/

#pragma once

#include <fg/defines.h>
#include <common.hpp>
#include <shader_headers/marker2d_vs.hpp>
#include <shader_headers/marker_fs.hpp>
#include <shader_headers/histogram_fs.hpp>
#include <shader_headers/plot3_vs.hpp>
#include <shader_headers/plot3_fs.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <memory>
#include <map>

namespace internal
{

class plot_impl : public AbstractRenderable {
    protected:
        GLuint    mDimension;
        /* plot points characteristics */
        GLuint    mNumPoints;
        fg::dtype mDataType;
        GLenum    mGLType;
        fg::MarkerType mMarkerType;
        fg::PlotType   mPlotType;
        /* OpenGL Objects */
        GLuint    mPlotProgram;
        GLuint    mMarkerProgram;
        /* shader variable index locations */
        GLuint    mPlotMatIndex;
        GLuint    mPlotPVCOnIndex;
        GLuint    mPlotPVAOnIndex;
        GLuint    mPlotUColorIndex;
        GLuint    mPlotRangeIndex;
        GLuint    mPlotPointIndex;
        GLuint    mPlotColorIndex;
        GLuint    mPlotAlphaIndex;

        GLuint    mMarkerPVCOnIndex;
        GLuint    mMarkerPVAOnIndex;
        GLuint    mMarkerTypeIndex;
        GLuint    mMarkerColIndex;
        GLuint    mMarkerMatIndex;
        GLuint    mMarkerPointIndex;
        GLuint    mMarkerColorIndex;
        GLuint    mMarkerAlphaIndex;

        std::map<int, GLuint> mVAOMap;

        /* bind and unbind helper functions
         * for rendering resources */
        void bindResources(const int pWindowId);
        void unbindResources() const;

        virtual void computeTransformMat(glm::mat4& pOut, const glm::mat4 pInput,
                                         const int pX, const int pY,
                                         const int pVPW, const int pVPH);
        virtual void bindDimSpecificUniforms(); // has to be called only after shaders are bound

    public:
        plot_impl(const uint pNumPoints, const fg::dtype pDataType,
                  const fg::PlotType pPlotType, const fg::MarkerType pMarkerType,
                  const int pDimension=3);
        ~plot_impl();

        virtual void render(const int pWindowId,
                            const int pX, const int pY, const int pVPW, const int pVPH,
                            const glm::mat4& pTransform);
};

class plot2d_impl : public plot_impl {
    protected:
        void computeTransformMat(glm::mat4& pOut, const glm::mat4 pInput,
                                 const int pX, const int pY,
                                 const int pVPW, const int pVPH) override;
        void bindDimSpecificUniforms() override; // has to be called only after shaders are bound

    public:
        plot2d_impl(const uint pNumPoints, const fg::dtype pDataType,
                    const fg::PlotType pPlotType, const fg::MarkerType pMarkerType)
            : plot_impl(pNumPoints, pDataType, pPlotType, pMarkerType, 2) {}
};

class _Plot {
    private:
        std::shared_ptr<plot_impl> mPlot;

    public:
        _Plot(const uint pNumPoints, const fg::dtype pDataType,
              const fg::PlotType pPlotType, const fg::MarkerType pMarkerType,
              const fg::ChartType pChartType) {
            if (pChartType == fg::FG_2D) {
                mPlot = std::make_shared< plot2d_impl >(pNumPoints, pDataType,
                        pPlotType, pMarkerType);
            } else {
                mPlot = std::make_shared< plot_impl >(pNumPoints, pDataType,
                        pPlotType, pMarkerType);
            }
        }

        inline const std::shared_ptr<plot_impl>& impl() const {
            return mPlot;
        }

        inline void setColor(const float pRed, const float pGreen,
                             const float pBlue, const float pAlpha) {
            mPlot->setColor(pRed, pGreen, pBlue, pAlpha);
        }

        inline void setLegend(const std::string pLegend) {
            mPlot->setLegend(pLegend);
        }

        inline GLuint vbo() const {
            return mPlot->vbo();
        }

        inline GLuint cbo() const {
            return mPlot->cbo();
        }

        inline GLuint abo() const {
            return mPlot->abo();
        }

        inline size_t vboSize() const {
            return mPlot->vboSize();
        }

        inline size_t cboSize() const {
            return mPlot->cboSize();
        }

        inline size_t aboSize() const {
            return mPlot->aboSize();
        }

        inline void render(const int pWindowId,
                           const int pX, const int pY, const int pVPW, const int pVPH,
                           const glm::mat4& pTransform) const {
            mPlot->render(pWindowId, pX, pY, pVPW, pVPH, pTransform);
        }
};

}

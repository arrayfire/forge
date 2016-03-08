/*******************************************************
* Copyright (c) 2015-2019, ArrayFire
* All rights reserved.
*
* This file is distributed under 3-clause BSD license.
* The complete license agreement can be obtained at:
* http://arrayfire.com/licenses/BSD-3-Clause
********************************************************/

#pragma once

#include <backend.hpp>
#include <plot.hpp>

#include <memory>

namespace common
{

class Plot {
    private:
        std::shared_ptr<detail::plot_impl> mPlot;

    public:
        Plot(const uint pNumPoints, const fg::dtype pDataType,
              const fg::PlotType pPlotType, const fg::MarkerType pMarkerType,
              const fg::ChartType pChartType) {
            if (pChartType == FG_2D) {
                mPlot = std::make_shared< detail::plot2d_impl >(pNumPoints, pDataType,
                        pPlotType, pMarkerType);
            } else {
                mPlot = std::make_shared< detail::plot_impl >(pNumPoints, pDataType,
                        pPlotType, pMarkerType);
            }
        }

        Plot(const fg_plot pOther) {
            mPlot = reinterpret_cast<Plot*>(pOther)->impl();
        }

        inline const std::shared_ptr<detail::plot_impl>& impl() const {
            return mPlot;
        }

        inline void setColor(const float pRed, const float pGreen,
                             const float pBlue, const float pAlpha) {
            mPlot->setColor(pRed, pGreen, pBlue, pAlpha);
        }

        inline void setLegend(const char* pLegend) {
            mPlot->setLegend(pLegend);
        }

        inline void setMarkerSize(const float pMarkerSize) {
            mPlot->setMarkerSize(pMarkerSize);
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

        inline GLuint mbo() const {
            return mPlot->markers();
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

        inline size_t mboSize() const {
            return mPlot->markersSizes();
        }

        inline void render(const int pWindowId,
                           const int pX, const int pY, const int pVPW, const int pVPH,
                           const glm::mat4& pTransform) const {
            mPlot->render(pWindowId, pX, pY, pVPW, pVPH, pTransform);
        }
};

}

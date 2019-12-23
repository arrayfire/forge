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
#include <common/chart_common.hpp>
#include <histogram_impl.hpp>
#include <plot_impl.hpp>
#include <surface_impl.hpp>
#include <vector_field_impl.hpp>

#include <memory>

namespace forge {
namespace common {

class Histogram : public ChartRenderableBase<detail::histogram_impl> {
   public:
    Histogram(unsigned pNBins, forge::dtype pDataType)
        : ChartRenderableBase<detail::histogram_impl>(
              std::make_shared<detail::histogram_impl>(pNBins, pDataType)) {}

    Histogram(const fg_histogram pOther)
        : ChartRenderableBase<detail::histogram_impl>(
              reinterpret_cast<Histogram*>(pOther)->impl()) {}
};

class Plot : public ChartRenderableBase<detail::plot_impl> {
   public:
    Plot(const unsigned pNumPoints, const forge::dtype pDataType,
         const forge::PlotType pPlotType, const forge::MarkerType pMarkerType,
         const forge::ChartType pChartType) {
        if (pChartType == FG_CHART_2D) {
            mShrdPtr = std::make_shared<detail::plot2d_impl>(
                pNumPoints, pDataType, pPlotType, pMarkerType);
        } else {
            mShrdPtr = std::make_shared<detail::plot_impl>(
                pNumPoints, pDataType, pPlotType, pMarkerType);
        }
    }

    Plot(const fg_plot pOther)
        : ChartRenderableBase<detail::plot_impl>(
              reinterpret_cast<Plot*>(pOther)->impl()) {}

    inline void setMarkerSize(const float pMarkerSize) {
        mShrdPtr->setMarkerSize(pMarkerSize);
    }

    inline unsigned mbo() const { return mShrdPtr->markers(); }

    inline size_t mboSize() const { return mShrdPtr->markersSizes(); }
};

class Surface : public ChartRenderableBase<detail::surface_impl> {
   public:
    Surface(const unsigned pNumXPoints, const unsigned pNumYPoints,
            const forge::dtype pDataType,
            const forge::PlotType pPlotType     = FG_PLOT_SURFACE,
            const forge::MarkerType pMarkerType = FG_MARKER_NONE) {
        switch (pPlotType) {
            case (FG_PLOT_SURFACE):
                mShrdPtr = std::make_shared<detail::surface_impl>(
                    pNumXPoints, pNumYPoints, pDataType, pMarkerType);
                break;
            case (FG_PLOT_SCATTER):
                mShrdPtr = std::make_shared<detail::scatter3_impl>(
                    pNumXPoints, pNumYPoints, pDataType, pMarkerType);
                break;
            default:
                mShrdPtr = std::make_shared<detail::surface_impl>(
                    pNumXPoints, pNumYPoints, pDataType, pMarkerType);
        };
    }

    Surface(const fg_surface pOther)
        : ChartRenderableBase<detail::surface_impl>(
              reinterpret_cast<Surface*>(pOther)->impl()) {}
};

class VectorField : public ChartRenderableBase<detail::vector_field_impl> {
   public:
    VectorField(const unsigned pNumPoints, const forge::dtype pDataType,
                const forge::ChartType pChartType) {
        if (pChartType == FG_CHART_2D) {
            mShrdPtr = std::make_shared<detail::vector_field2d_impl>(pNumPoints,
                                                                     pDataType);
        } else {
            mShrdPtr = std::make_shared<detail::vector_field_impl>(pNumPoints,
                                                                   pDataType);
        }
    }

    VectorField(const fg_vector_field pOther)
        : ChartRenderableBase<detail::vector_field_impl>(
              reinterpret_cast<VectorField*>(pOther)->impl()) {}

    inline unsigned dbo() const { return mShrdPtr->directions(); }

    inline size_t dboSize() const { return mShrdPtr->directionsSize(); }
};

}  // namespace common
}  // namespace forge

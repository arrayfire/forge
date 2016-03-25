/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fg/chart.h>
#include <fg/exception.h>
#include <fg/font.h>
#include <fg/histogram.h>
#include <fg/image.h>
#include <fg/plot.h>
#include <fg/surface.h>
#include <fg/window.h>

#include <handle.hpp>
#include <Chart.hpp>
#include <chart_renderables.hpp>
#include <Font.hpp>
#include <Image.hpp>
#include <Window.hpp>

fg_err fg_create_chart(fg_chart *pHandle,
                       const fg_chart_type pChartType)
{
    try {
        *pHandle = getHandle(new common::Chart(pChartType));
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_destroy_chart(fg_chart pHandle)
{
    try {
        delete getChart(pHandle);
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_set_chart_axes_titles(fg_chart pHandle,
                                const char* pX,
                                const char* pY,
                                const char* pZ)
{
    try {
        getChart(pHandle)->setAxesTitles(pX, pY, pZ);
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_set_chart_axes_limits(fg_chart pHandle,
                                const float pXmin, const float pXmax,
                                const float pYmin, const float pYmax,
                                const float pZmin, const float pZmax)
{
    try {
        getChart(pHandle)->setAxesLimits(pXmin, pXmax, pYmin, pYmax, pZmin, pZmax);
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_set_chart_legend_position(fg_chart pHandle, const float pX, const float pY)
{
    try {
        getChart(pHandle)->setLegendPosition(pX, pY);
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_add_image_to_chart(fg_image* pImage, fg_chart pHandle,
                             const uint pWidth, const uint pHeight,
                             const fg_channel_format pFormat,
                             const fg_dtype pType)
{
    try {
        common::Image* img = new common::Image(pWidth, pHeight, pFormat, pType);
        getChart(pHandle)->addRenderable(img->impl());
        *pImage = getHandle(img);
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_add_histogram_to_chart(fg_histogram* pHistogram, fg_chart pHandle,
                                 const uint pNBins, const fg_dtype pType)
{
    try {
        common::Chart* chrt = getChart(pHandle);

        if (chrt->chartType()== FG_CHART_2D) {
            common::Histogram* hist = new common::Histogram(pNBins, pType);
            chrt->addRenderable(hist->impl());
            *pHistogram = getHandle(hist);
        } else {
            throw fg::ArgumentError("Chart::render", __LINE__, 5,
                    "Can add histogram to a 2d chart only");
        }
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_add_plot_to_chart(fg_plot* pPlot, fg_chart pHandle,
                            const uint pNPoints, const fg_dtype pType,
                            const fg_plot_type pPlotType, const fg_marker_type pMarkerType)
{
    try {
        common::Chart* chrt = getChart(pHandle);
        fg::ChartType ctype = chrt->chartType();

        if (ctype == FG_CHART_2D) {
            common::Plot* plt = new common::Plot(pNPoints, pType, pPlotType, pMarkerType, FG_CHART_2D);
            chrt->addRenderable(plt->impl());
            *pPlot = getHandle(plt);
        } else {
            common::Plot* plt = new common::Plot(pNPoints, pType, pPlotType, pMarkerType, FG_CHART_3D);
            chrt->addRenderable(plt->impl());
            *pPlot = getHandle(plt);
        }
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_add_surface_to_chart(fg_surface* pSurface, fg_chart pHandle,
                               const uint pXPoints, const uint pYPoints, const fg_dtype pType,
                               const fg_plot_type pPlotType, const fg_marker_type pMarkerType)
{
    try {
        common::Chart* chrt = getChart(pHandle);
        fg::ChartType ctype = chrt->chartType();

        if (ctype == FG_CHART_3D) {
            common::Surface* surf = new common::Surface(pXPoints, pYPoints, pType,
                                                        pPlotType, pMarkerType);
            chrt->addRenderable(surf->impl());
            *pSurface = getHandle(surf);
        } else {
            throw fg::ArgumentError("Chart::render", __LINE__, 5,
                    "Can add surface plot to a 3d chart only");
        }
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_render_chart(const fg_window pWindow, const fg_chart pChart,
                       const int pX, const int pY, const int pWidth, const int pHeight,
                       const float* pTransform)
{
    try {
        getChart(pChart)->render(getWindow(pWindow)->getID(),
                                 pX, pY, pWidth, pHeight,
                                 glm::make_mat4(pTransform));
    }
    CATCHALL

    return FG_ERR_NONE;
}

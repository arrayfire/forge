/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fg/surface.h>

#include <handle.hpp>
#include <chart_renderables.hpp>

using namespace forge;

fg_err fg_create_surface(fg_surface *pSurface,
                      const uint pXPoints, const uint pYPoints,
                      const fg_dtype pType,
                      const fg_plot_type pPlotType,
                      const fg_marker_type pMarkerType)
{
    try {
        *pSurface = getHandle(new common::Surface(pXPoints, pYPoints, (forge::dtype)pType,
                                                  pPlotType, pMarkerType));
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_destroy_surface(fg_surface pSurface)
{
    try {
        delete getSurface(pSurface);
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_set_surface_color(fg_surface pSurface,
                            const float pRed, const float pGreen,
                            const float pBlue, const float pAlpha)
{
    try {
        getSurface(pSurface)->setColor(pRed, pGreen, pBlue, pAlpha);
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_set_surface_legend(fg_surface pSurface, const char* pLegend)
{
    try {
        getSurface(pSurface)->setLegend(pLegend);
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_surface_vbo(uint* pOut, const fg_surface pSurface)
{
    try {
        *pOut = getSurface(pSurface)->vbo();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_surface_cbo(uint* pOut, const fg_surface pSurface)
{
    try {
        *pOut = getSurface(pSurface)->cbo();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_surface_abo(uint* pOut, const fg_surface pSurface)
{
    try {
        *pOut = getSurface(pSurface)->abo();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_surface_vbo_size(uint* pOut, const fg_surface pSurface)
{
    try {
        *pOut = (uint)getSurface(pSurface)->vboSize();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_surface_cbo_size(uint* pOut, const fg_surface pSurface)
{
    try {
        *pOut = (uint)getSurface(pSurface)->cboSize();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_surface_abo_size(uint* pOut, const fg_surface pSurface)
{
    try {
        *pOut = (uint)getSurface(pSurface)->aboSize();
    }
    CATCHALL

    return FG_ERR_NONE;
}

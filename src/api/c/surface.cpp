/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/chart_renderables.hpp>
#include <common/handle.hpp>
#include <fg/surface.h>

using namespace forge;

using forge::common::getSurface;

fg_err fg_create_surface(fg_surface* pSurface, const unsigned pXPoints,
                         const unsigned pYPoints, const fg_dtype pType,
                         const fg_plot_type pPlotType,
                         const fg_marker_type pMarkerType) {
    try {
        ARG_ASSERT(1, (pXPoints > 0));
        ARG_ASSERT(2, (pYPoints > 0));

        *pSurface = getHandle(new common::Surface(
            pXPoints, pYPoints, (forge::dtype)pType, pPlotType, pMarkerType));
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_retain_surface(fg_surface* pOut, fg_surface pIn) {
    try {
        ARG_ASSERT(1, (pIn != 0));

        common::Surface* temp = new common::Surface(getSurface(pIn));
        *pOut                 = getHandle(temp);
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_release_surface(fg_surface pSurface) {
    try {
        ARG_ASSERT(0, (pSurface != 0));

        delete getSurface(pSurface);
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_set_surface_color(fg_surface pSurface, const float pRed,
                            const float pGreen, const float pBlue,
                            const float pAlpha) {
    try {
        ARG_ASSERT(0, (pSurface != 0));

        getSurface(pSurface)->setColor(pRed, pGreen, pBlue, pAlpha);
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_set_surface_legend(fg_surface pSurface, const char* pLegend) {
    try {
        ARG_ASSERT(0, (pSurface != 0));
        ARG_ASSERT(1, (pLegend != 0));

        getSurface(pSurface)->setLegend(pLegend);
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_surface_vertex_buffer(unsigned* pOut, const fg_surface pSurface) {
    try {
        ARG_ASSERT(1, (pSurface != 0));

        *pOut = getSurface(pSurface)->vbo();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_surface_color_buffer(unsigned* pOut, const fg_surface pSurface) {
    try {
        ARG_ASSERT(1, (pSurface != 0));

        *pOut = getSurface(pSurface)->cbo();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_surface_alpha_buffer(unsigned* pOut, const fg_surface pSurface) {
    try {
        ARG_ASSERT(1, (pSurface != 0));

        *pOut = getSurface(pSurface)->abo();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_surface_vertex_buffer_size(unsigned* pOut,
                                         const fg_surface pSurface) {
    try {
        ARG_ASSERT(1, (pSurface != 0));

        *pOut = (unsigned)getSurface(pSurface)->vboSize();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_surface_color_buffer_size(unsigned* pOut,
                                        const fg_surface pSurface) {
    try {
        ARG_ASSERT(1, (pSurface != 0));

        *pOut = (unsigned)getSurface(pSurface)->cboSize();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_surface_alpha_buffer_size(unsigned* pOut,
                                        const fg_surface pSurface) {
    try {
        ARG_ASSERT(1, (pSurface != 0));

        *pOut = (unsigned)getSurface(pSurface)->aboSize();
    }
    CATCHALL

    return FG_ERR_NONE;
}

/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fg/pie.h>

#include <common/chart_renderables.hpp>
#include <common/handle.hpp>

using namespace forge;

using forge::common::getPie;

fg_err fg_create_pie(fg_pie* pPie, const unsigned pNSectors,
                     const fg_dtype pType) {
    try {
        ARG_ASSERT(1, (pNSectors > 0));

        *pPie = getHandle(new common::Pie(pNSectors, (forge::dtype)pType));
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_retain_pie(fg_pie* pOut, fg_pie pIn) {
    try {
        ARG_ASSERT(1, (pIn != 0));

        common::Pie* temp = new common::Pie(getPie(pIn));
        *pOut             = getHandle(temp);
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_release_pie(fg_pie pPie) {
    try {
        ARG_ASSERT(0, (pPie != 0));

        delete getPie(pPie);
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_set_pie_color(fg_pie pPie, const float pRed, const float pGreen,
                        const float pBlue, const float pAlpha) {
    try {
        ARG_ASSERT(0, (pPie != 0));

        getPie(pPie)->setColor(pRed, pGreen, pBlue, pAlpha);
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_set_pie_legend(fg_pie pPie, const char* pLegend) {
    try {
        ARG_ASSERT(0, (pPie != 0));
        ARG_ASSERT(1, (pLegend != 0));

        getPie(pPie)->setLegend(pLegend);
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_pie_vertex_buffer(unsigned* pOut, const fg_pie pPie) {
    try {
        ARG_ASSERT(1, (pPie != 0));

        *pOut = getPie(pPie)->vbo();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_pie_color_buffer(unsigned* pOut, const fg_pie pPie) {
    try {
        ARG_ASSERT(1, (pPie != 0));

        *pOut = getPie(pPie)->cbo();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_pie_alpha_buffer(unsigned* pOut, const fg_pie pPie) {
    try {
        ARG_ASSERT(1, (pPie != 0));

        *pOut = getPie(pPie)->abo();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_pie_vertex_buffer_size(unsigned* pOut, const fg_pie pPie) {
    try {
        ARG_ASSERT(1, (pPie != 0));

        *pOut = (unsigned)getPie(pPie)->vboSize();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_pie_color_buffer_size(unsigned* pOut, const fg_pie pPie) {
    try {
        ARG_ASSERT(1, (pPie != 0));

        *pOut = (unsigned)getPie(pPie)->cboSize();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_pie_alpha_buffer_size(unsigned* pOut, const fg_pie pPie) {
    try {
        ARG_ASSERT(1, (pPie != 0));

        *pOut = (unsigned)getPie(pPie)->aboSize();
    }
    CATCHALL

    return FG_ERR_NONE;
}

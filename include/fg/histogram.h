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

namespace internal
{
class _Histogram;
}
namespace fg
{

class Histogram {
    private:
        internal::_Histogram* value;

    public:
        FGAPI Histogram(GLuint pNBins, GLenum pDataType);
        FGAPI ~Histogram();

        FGAPI void setBarColor(float r, float g, float b);
        FGAPI void setAxesLimits(float pXmax, float pXmin, float pYmax, float pYmin);
        FGAPI void setXAxisTitle(const char* pTitle);
        FGAPI void setYAxisTitle(const char* pTitle);

        FGAPI float xmax() const;
        FGAPI float xmin() const;
        FGAPI float ymax() const;
        FGAPI float ymin() const;

        FGAPI GLuint vbo() const;
        FGAPI size_t size() const;
        FGAPI internal::_Histogram* get() const;

        FGAPI void render(int pX, int pY, int pViewPortWidth, int pViewPortHeight) const;
};

}

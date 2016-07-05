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


#ifdef __cplusplus
extern "C" {
#endif

FGAPI fg_err fg_create_vector_field(fg_vector_field *pField,
                                    const uint pNPoints,
                                    const fg_dtype pType,
                                    const fg_chart_type pChartType);

FGAPI fg_err fg_destroy_vector_field(fg_vector_field pField);

FGAPI fg_err fg_set_vector_field_color(fg_vector_field pField,
                                       const float pRed, const float pGreen,
                                       const float pBlue, const float pAlpha);

FGAPI fg_err fg_set_vector_field_legend(fg_vector_field pField, const char* pLegend);

FGAPI fg_err fg_get_vector_field_vbo(uint* pOut, const fg_vector_field pField);

FGAPI fg_err fg_get_vector_field_cbo(uint* pOut, const fg_vector_field pField);

FGAPI fg_err fg_get_vector_field_abo(uint* pOut, const fg_vector_field pField);

FGAPI fg_err fg_get_vector_field_dbo(uint* pOut, const fg_vector_field pField);

FGAPI fg_err fg_get_vector_field_vbo_size(uint* pOut, const fg_vector_field pField);

FGAPI fg_err fg_get_vector_field_cbo_size(uint* pOut, const fg_vector_field pField);

FGAPI fg_err fg_get_vector_field_abo_size(uint* pOut, const fg_vector_field pField);

FGAPI fg_err fg_get_vector_field_dbo_size(uint* pOut, const fg_vector_field pField);

#ifdef __cplusplus
}
#endif


#ifdef __cplusplus

namespace fg
{

/**
   \class VectorField

   \brief VectorField is a line graph to display two dimensional data.
 */
class VectorField {
    private:
        fg_vector_field mValue;

    public:
        /**
           Creates a VectorField object

           \param[in] pNumPoints is number of data points to display
           \param[in] pDataType takes one of the values of \ref dtype that indicates
                      the integral data type of VectorField data
           \param[in] pChartType dictates the dimensionality of the chart
         */
        FGAPI VectorField(const uint pNumPoints, const dtype pDataType, const ChartType pChartType);

        /**
           Copy constructor for VectorField

           \param[in] pOther is the VectorField of which we make a copy of.
         */
        FGAPI VectorField(const VectorField& pOther);

        /**
           VectorField Destructor
         */
        FGAPI ~VectorField();

        /**
           Set global color for the field lines

           \param[in] pColor takes values of fg::Color to define VectorField color
        */
        FGAPI void setColor(const fg::Color pColor);

        /**
           Set global color for the field lines

           \param[in] pRed is Red component in range [0, 1]
           \param[in] pGreen is Green component in range [0, 1]
           \param[in] pBlue is Blue component in range [0, 1]
           \param[in] pAlpha is Blue component in range [0, 1]
         */
        FGAPI void setColor(const float pRed, const float pGreen,
                            const float pBlue, const float pAlpha);

        /**
           Set VectorField legend

           \param[in] pLegend
         */
        FGAPI void setLegend(const char* pLegend);

        /**
           Get the OpenGL buffer object identifier for vertices

           \return OpenGL VBO resource id.
         */
        FGAPI uint vertices() const;

        /**
           Get the OpenGL buffer object identifier for color values per vertex

           \return OpenGL CBO resource id.
         */
        FGAPI uint colors() const;

        /**
           Get the OpenGL buffer object identifier for alpha values per vertex

           \return OpenGL ABO resource id.
         */
        FGAPI uint alphas() const;

        /**
           Get the OpenGL buffer object identifier for field direction per vertex

           \return OpenGL DBO resource id.
         */
        FGAPI uint directions() const;

        /**
           Get the OpenGL Vertex Buffer Object resource size

           \return vertex buffer object size in bytes
         */
        FGAPI uint verticesSize() const;

        /**
           Get the OpenGL colors Buffer Object resource size

           \return colors buffer object size in bytes
         */
        FGAPI uint colorsSize() const;

        /**
           Get the OpenGL alphas Buffer Object resource size

           \return alphas buffer object size in bytes
         */
        FGAPI uint alphasSize() const;

        /**
           Get the OpenGL directions Buffer Object resource size

           \return direction buffer object size in bytes
         */
        FGAPI uint directionsSize() const;

        /**
           Get the handle to internal implementation of VectorField
         */
        FGAPI fg_vector_field get() const;
};

}

#endif

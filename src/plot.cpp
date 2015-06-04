/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fg/plot.h>
#include <plot.hpp>
#include <common.hpp>

#include <cmath>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

using namespace std;

namespace internal
{

void plot_impl::bindResources() const
{
    // attach plot vertices
    glEnableVertexAttribArray(mPointIndex);
    glBindBuffer(GL_ARRAY_BUFFER, mMainVBO);
    glVertexAttribPointer(mPointIndex, 2, mDataType, GL_FALSE, 0, 0);
}

void plot_impl::unbindResources() const
{
    glDisableVertexAttribArray(mPointIndex);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

plot_impl::plot_impl(unsigned pNumPoints, fg::FGType pDataType)
    : AbstractChart2D(), mNumPoints(pNumPoints), mDataType(FGTypeToGLenum(pDataType)),
      mMainVBO(0), mMainVBOsize(0), mPointIndex(0)
{
    unsigned total_points = 2*mNumPoints;
    // buffersubdata calls on mMainVBO
    // will only update the points data
    switch(mDataType) {
        case GL_FLOAT:
            mMainVBO = createBuffer<float>(GL_ARRAY_BUFFER, total_points, NULL, GL_DYNAMIC_DRAW);
            mMainVBOsize = total_points*sizeof(float);
            break;
        case GL_INT:
            mMainVBO = createBuffer<int>(GL_ARRAY_BUFFER, total_points, NULL, GL_DYNAMIC_DRAW);
            mMainVBOsize = total_points*sizeof(int);
            break;
        case GL_UNSIGNED_INT:
            mMainVBO = createBuffer<unsigned>(GL_ARRAY_BUFFER, total_points, NULL, GL_DYNAMIC_DRAW);
            mMainVBOsize = total_points*sizeof(unsigned);
            break;
        case GL_UNSIGNED_BYTE:
            mMainVBO = createBuffer<unsigned char>(GL_ARRAY_BUFFER, total_points, NULL, GL_DYNAMIC_DRAW);
            mMainVBOsize = total_points*sizeof(unsigned char);
            break;
        default: fg::TypeError("Plot::Plot", __LINE__, 1, GLenumToFGType(mDataType));
    }
    mPointIndex = borderProgramPointIndex();
}

plot_impl::~plot_impl()
{
    CheckGL("Begin Plot::~Plot");
    glDeleteBuffers(1, &mMainVBO);
    CheckGL("End Plot::~Plot");
}

void plot_impl::setColor(float r, float g, float b)
{
    mLineColor[0] = clampTo01(r);
    mLineColor[1] = clampTo01(g);
    mLineColor[2] = clampTo01(b);
    mLineColor[3] = 1.0f;
}

GLuint plot_impl::vbo() const
{
    return mMainVBO;
}

size_t plot_impl::size() const
{
    return mMainVBOsize;
}

void plot_impl::render(int pX, int pY, int pVPW, int pVPH) const
{
    float graph_scale_x = 1/(xmax() - xmin());
    float graph_scale_y = 1/(ymax() - ymin());

    CheckGL("Begin Plot::render");
    /* Enavle scissor test to discard anything drawn beyond viewport.
     * Set scissor rectangle to clip fragments outside of viewport */
    glScissor(pX+leftMargin()+tickSize(), pY+bottomMargin()+tickSize(),
            pVPW - (leftMargin()+rightMargin()+tickSize()),
            pVPH - (bottomMargin()+topMargin()+tickSize()));
    glEnable(GL_SCISSOR_TEST);

    bindBorderProgram();
    glm::mat4 transform = glm::scale(glm::mat4(1.0f), glm::vec3(graph_scale_x, graph_scale_y, 1));
    glUniformMatrix4fv(borderMatIndex(), 1, GL_FALSE, glm::value_ptr(transform));
    glUniform4fv(borderColorIndex(), 1, mLineColor);

    /* render the plot data */
    bindResources();
    glDrawArrays(GL_LINE_STRIP, 0, mNumPoints);
    unbindResources();

    /* Stop clipping and reset viewport to window dimensions */
    glDisable(GL_SCISSOR_TEST);
    unbindBorderProgram();

    /* render graph border and axes */
    renderChart(pX, pY, pVPW, pVPH);

    CheckGL("End Plot::render");
}

}

namespace fg
{

Plot::Plot(unsigned pNumPoints, fg::FGType pDataType)
{
    value = new internal::_Plot(pNumPoints, pDataType);
}

Plot::Plot(const Plot& other)
{
    value = new internal::_Plot(*other.get());
}

Plot::~Plot()
{
    delete value;
}

void Plot::setColor(float r, float g, float b)
{
    value->setColor(r, g, b);
}

void Plot::setAxesLimits(float pXmax, float pXmin, float pYmax, float pYmin)
{
    value->setAxesLimits(pXmax, pXmin, pYmax, pYmin);
}

void Plot::setXAxisTitle(const char* pTitle)
{
    value->setXAxisTitle(pTitle);
}

void Plot::setYAxisTitle(const char* pTitle)
{
    value->setYAxisTitle(pTitle);
}

float Plot::xmax() const
{
    return value->xmax();
}

float Plot::xmin() const
{
    return value->xmin();
}

float Plot::ymax() const
{
    return value->ymax();
}

float Plot::ymin() const
{
    return value->ymin();
}

unsigned Plot::vbo() const
{
    return value->vbo();
}

unsigned Plot::size() const
{
    return value->size();
}

internal::_Plot* Plot::get() const
{
    return value;
}

void Plot::render(int pX, int pY, int pViewPortWidth, int pViewPortHeight) const
{
    value->render(pX, pY, pViewPortWidth, pViewPortHeight);
}

}

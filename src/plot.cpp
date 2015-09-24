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

const char *gMarkerVertexShaderSrc =
"#version 330\n"
"in vec2 point;\n"
"uniform mat4 transform;\n"
"void main(void) {\n"
"   gl_Position = transform * vec4(point.xy, 0, 1);\n"
"   gl_PointSize = 10;\n"
"}";


const char *gMarkerSpriteFragmentShaderSrc =
"#version 330\n"
"uniform int marker_type;\n"
"uniform vec4 line_color;\n"
"out vec4 outputColor;\n"
"void main(void) {\n"
"   float dist = sqrt( (gl_PointCoord.x - 0.5) * (gl_PointCoord.x-0.5) + (gl_PointCoord.y-0.5) * (gl_PointCoord.y-0.5) );\n"
"   bool in_bounds;\n"
"   switch(marker_type) {\n"
"       case 1:\n"
"           in_bounds = dist < 0.3;\n"
"           break;\n"
"       case 2:\n"
"           in_bounds = ( (dist > 0.3) && (dist<0.5) );\n"
"           break;\n"
"       case 3:\n"
"           in_bounds = ((gl_PointCoord.x < 0.15) || (gl_PointCoord.x > 0.85)) ||\n"
"                       ((gl_PointCoord.y < 0.15) || (gl_PointCoord.y > 0.85));\n"
"           break;\n"
"       case 4:\n"
"           in_bounds = (2*(gl_PointCoord.x - 0.25) - (gl_PointCoord.y + 0.5) < 0) && (2*(gl_PointCoord.x - 0.25) + (gl_PointCoord.y + 0.5) > 1);\n"
"           break;\n"
"       case 5:\n"
"           in_bounds = abs((gl_PointCoord.x - 0.5) + (gl_PointCoord.y - 0.5) ) < 0.13  ||\n"
"           abs((gl_PointCoord.x - 0.5) - (gl_PointCoord.y - 0.5) ) < 0.13  ;\n"
"           break;\n"
"       case 6:\n"
"           in_bounds = abs((gl_PointCoord.x - 0.5)) < 0.07 ||\n"
"           abs((gl_PointCoord.y - 0.5)) < 0.07;\n"
"           break;\n"
"       case 7:\n"
"           in_bounds = abs((gl_PointCoord.x - 0.5) + (gl_PointCoord.y - 0.5) ) < 0.07 ||\n"
"           abs((gl_PointCoord.x - 0.5) - (gl_PointCoord.y - 0.5) ) < 0.07 ||\n"
"           abs((gl_PointCoord.x - 0.5)) < 0.07 ||\n"
"           abs((gl_PointCoord.y - 0.5)) < 0.07;\n"
"           break;\n"
"       default:\n"
"           in_bounds = true;\n"
"   }\n"
"   if(!in_bounds)\n"
"       discard;\n"
"   else\n"
"       outputColor = line_color;\n"
"}";

namespace internal
{

void plot_impl::bindResources(int pWindowId)
{
    if (mVAOMap.find(pWindowId) == mVAOMap.end()) {
        GLuint vao = 0;
        /* create a vertex array object
         * with appropriate bindings */
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        // attach plot vertices
        glEnableVertexAttribArray(mPointIndex);
        glBindBuffer(GL_ARRAY_BUFFER, mMainVBO);
        glVertexAttribPointer(mPointIndex, 2, mGLType, GL_FALSE, 0, 0);
        glBindVertexArray(0);
        /* store the vertex array object corresponding to
         * the window instance in the map */
        mVAOMap[pWindowId] = vao;
    }

    glBindVertexArray(mVAOMap[pWindowId]);
}

void plot_impl::unbindResources() const
{
    glBindVertexArray(0);
}

plot_impl::plot_impl(unsigned pNumPoints, fg::dtype pDataType, fg::MarkerType pMarkerType)
    : AbstractChart2D(), mNumPoints(pNumPoints),
      mDataType(pDataType), mGLType(gl_dtype(mDataType)),
      mMainVBO(0), mMainVBOsize(0), mPointIndex(0)
{
    unsigned total_points = 2*mNumPoints;
    // buffersubdata calls on mMainVBO
    // will only update the points data
    switch(mGLType) {
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
        default: fg::TypeError("Plot::Plot", __LINE__, 1, mDataType);
    }
    mPointIndex = borderProgramPointIndex();
    mMarkerType = pMarkerType;
    mMarkerProgram = initShaders(gMarkerVertexShaderSrc, gMarkerSpriteFragmentShaderSrc);

    mMarkerTypeIndex = glGetUniformLocation(mMarkerProgram, "marker_type");
    mSpriteTMatIndex  = glGetUniformLocation(mMarkerProgram, "transform");
}

plot_impl::~plot_impl()
{
    CheckGL("Begin Plot::~Plot");
    glDeleteBuffers(1, &mMainVBO);
    CheckGL("End Plot::~Plot");
}

void plot_impl::setColor(fg::Color col)
{
    mLineColor[0] = (((int) col >> 24 ) & 0xFF ) / 255.f;
    mLineColor[1] = (((int) col >> 16 ) & 0xFF ) / 255.f;
    mLineColor[2] = (((int) col >> 8  ) & 0xFF ) / 255.f;
    mLineColor[3] = (((int) col       ) & 0xFF ) / 255.f;
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

void plot_impl::render(int pWindowId, int pX, int pY, int pVPW, int pVPH)
{
    float range_x = xmax() - xmin();
    float range_y = ymax() - ymin();
    // set scale to zero if input is constant array
    // otherwise compute scale factor by standard equation
    float graph_scale_x = std::abs(range_x) < 1.0e-3 ? 0.0f : 2/(xmax() - xmin());
    float graph_scale_y = std::abs(range_y) < 1.0e-3 ? 0.0f : 2/(ymax() - ymin());

    CheckGL("Begin Plot::render");
    float viewWidth    = pVPW - (leftMargin() + rightMargin() + tickSize()/2 );
    float viewHeight   = pVPH - (bottomMargin() + topMargin() + tickSize() );
    float view_scale_x = viewWidth/pVPW;
    float view_scale_y = viewHeight/pVPH;
    float view_offset_x = (2.0f * (leftMargin() + tickSize()/2 )/ pVPW ) ;
    float view_offset_y = (2.0f * (bottomMargin() + tickSize() )/ pVPH ) ;
    /* Enable scissor test to discard anything drawn beyond viewport.
     * Set scissor rectangle to clip fragments outside of viewport */
    glScissor(pX + leftMargin() + tickSize()/2, pY+bottomMargin() + tickSize()/2,
              pVPW - leftMargin()   - rightMargin() - tickSize()/2,
              pVPH - bottomMargin() - topMargin()   - tickSize()/2);
    glEnable(GL_SCISSOR_TEST);

    float coor_offset_x = ( -xmin() * graph_scale_x * view_scale_x);
    float coor_offset_y = ( -ymin() * graph_scale_y * view_scale_y);
    glm::mat4 transform = glm::translate(glm::mat4(1.f),
            glm::vec3(-1 + view_offset_x + coor_offset_x  , -1 + view_offset_y + coor_offset_y, 0));
    transform = glm::scale(transform,
            glm::vec3(graph_scale_x * view_scale_x , graph_scale_y * view_scale_y ,1));

    renderGraph(pWindowId, transform);

    /* Stop clipping and reset viewport to window dimensions */
    glDisable(GL_SCISSOR_TEST);
    /* render graph border and axes */
    renderChart(pWindowId, pX, pY, pVPW, pVPH);

    CheckGL("End Plot::render");
}

void plot_impl::renderGraph(int pWindowId, glm::mat4 transform){
    bindBorderProgram();
    glUniformMatrix4fv(borderMatIndex(), 1, GL_FALSE, glm::value_ptr(transform));
    glUniform4fv(borderColorIndex(), 1, mLineColor);
    bindResources(pWindowId);
    glDrawArrays(GL_LINE_STRIP, 0, mNumPoints);
    unbindResources();
    unbindBorderProgram();

    if(mMarkerType != fg::FG_NONE){
        glEnable(GL_PROGRAM_POINT_SIZE);
        glUseProgram(mMarkerProgram);

        glUniformMatrix4fv(spriteMatIndex(), 1, GL_FALSE, glm::value_ptr(transform));
        glUniform4fv(borderColorIndex(), 1, mLineColor);
        glUniform1i(markerTypeIndex(), mMarkerType);

        bindResources(pWindowId);
        glDrawArrays(GL_POINTS, 0, mNumPoints);
        unbindResources();
        glUseProgram(0);
        glDisable(GL_PROGRAM_POINT_SIZE);
    }
}

GLuint plot_impl::markerTypeIndex() const
{
    return mMarkerTypeIndex;
}

GLuint plot_impl::spriteMatIndex() const
{
    return mSpriteTMatIndex;
}


void scatter_impl::renderGraph(int pWindowId, glm::mat4 transform){
    if(mMarkerType != fg::FG_NONE){
        glEnable(GL_PROGRAM_POINT_SIZE);
        glUseProgram(mMarkerProgram);

        glUniformMatrix4fv(spriteMatIndex(), 1, GL_FALSE, glm::value_ptr(transform));
        glUniform4fv(borderColorIndex(), 1, mLineColor);
        glUniform1i(markerTypeIndex(), mMarkerType);

        bindResources(pWindowId);
        glDrawArrays(GL_POINTS, 0, mNumPoints);
        unbindResources();
        glUseProgram(0);
        glDisable(GL_PROGRAM_POINT_SIZE);
    }
}

}

namespace fg
{

Plot::Plot(unsigned pNumPoints, fg::dtype pDataType, fg::PlotType pPlotType, fg::MarkerType pMarkerType)
{
    value = new internal::_Plot(pNumPoints, pDataType, pPlotType, pMarkerType);
}

Plot::Plot(const Plot& other)
{
    value = new internal::_Plot(*other.get());
}

Plot::~Plot()
{
    delete value;
}

void Plot::setColor(fg::Color col)
{
    value->setColor(col);
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
    return (unsigned)value->size();
}

internal::_Plot* Plot::get() const
{
    return value;
}

}

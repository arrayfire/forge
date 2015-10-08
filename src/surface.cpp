/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fg/surface.h>
#include <surface.hpp>
#include <common.hpp>

#include <cmath>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

using namespace std;

static const char *gMarkerVertexShaderSrc =
"#version 330\n"
"in vec3 point;\n"
"uniform vec2 minmaxs[3];\n"
"out vec4 hpoint;\n"
"uniform mat4 transform;\n"
"void main(void) {\n"
"   gl_Position = transform * vec4(point.xyz, 1);\n"
"   hpoint=vec4(point.xyz,1);\n"
"   gl_PointSize=10;\n"
"}";

const char *gSurfFragmentShaderSrc =
"#version 330\n"
"uniform vec2 minmaxs[3];\n"
"in vec4 hpoint;\n"
"out vec4 outputColor;\n"
"vec3 hsv2rgb(vec3 c){\n"
"   vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);\n"
"   vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);\n"
"   return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);\n"
"}\n"
"void main(void) {\n"
"   bool nin_bounds = (hpoint.x > minmaxs[0].x || hpoint.x < minmaxs[0].y ||\n"
"       hpoint.y > minmaxs[1].x || hpoint.y < minmaxs[1].y || hpoint.z < minmaxs[2].y);\n"
"   float height = (minmaxs[2].x- hpoint.z)/(minmaxs[2].x-minmaxs[2].y);\n"
"   if(nin_bounds) discard;\n"
"   outputColor = vec4(hsv2rgb(vec3(height, 1.f, 1.f)),1);\n"
"}";

static const char *gMarkerSpriteFragmentShaderSrc =
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
"       case 8:\n"
"           in_bounds = true;\n"
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

void surface_impl::bindResources(int pWindowId)
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
        glVertexAttribPointer(mPointIndex, 3, mDataType, GL_FALSE, 0, 0);
        //attach indices
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mIndexVBO);
        glBindVertexArray(0);
        /* store the vertex array object corresponding to
         * the window instance in the map */
        mVAOMap[pWindowId] = vao;
    }

    glBindVertexArray(mVAOMap[pWindowId]);
}

void surface_impl::unbindResources() const
{
    glBindVertexArray(0);
}

void generate_grid_indices(unsigned short rows, unsigned short cols, unsigned short *indices){
    unsigned short idx = 0;
    for(unsigned short r = 0; r < rows-1; ++r){
        for(unsigned short c = 0; c < cols*2; ++c){
            unsigned short i = c + (r * (cols*2));

            if(c == cols * 2 - 1) {
                *indices++ = idx;
            }else{
                *indices++ = idx;
                if(i%2 == 0){
                    idx += cols;
                } else {
                    idx -= (r%2 == 0) ? (cols-1) : (cols+1);
                }
            }
        }
    }
}

surface_impl::surface_impl(unsigned pNumXPoints, unsigned pNumYPoints, fg::dtype pDataType, fg::MarkerType pMarkerType)
    : AbstractChart3D(), mNumXPoints(pNumXPoints),mNumYPoints(pNumYPoints), mDataType(gl_dtype(pDataType)),
      mMainVBO(0), mMainVBOsize(0), mIndexVBO(0), mIndexVBOsize(0), mPointIndex(0)
{
    unsigned total_points = 3*(mNumXPoints * mNumYPoints);

    mIndexVBOsize = (2 * mNumYPoints) * (mNumXPoints - 1);
    unsigned short indices[mIndexVBOsize];
    generate_grid_indices(mNumXPoints, mNumYPoints, indices);
    mIndexVBO = createBuffer<unsigned short>(GL_ELEMENT_ARRAY_BUFFER, mIndexVBOsize, NULL, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mIndexVBO);
    glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, mIndexVBOsize * sizeof(unsigned short), indices);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

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
        default: fg::TypeError("Plot::Plot", __LINE__, 1, pDataType);
    }
    mPointIndex = borderProgramPointIndex();
    mMarkerType = pMarkerType;
    mSurfProgram   = initShaders(gMarkerVertexShaderSrc, gSurfFragmentShaderSrc);
    mMarkerProgram = initShaders(gMarkerVertexShaderSrc, gMarkerSpriteFragmentShaderSrc);

    mSurfPointIndex   = glGetAttribLocation (mSurfProgram, "point");
    mSurfTMatIndex    = glGetUniformLocation(mSurfProgram, "transform");
    mSurfRangeIndex   = glGetUniformLocation(mSurfProgram, "minmaxs");

    mMarkerTypeIndex  = glGetUniformLocation(mMarkerProgram, "marker_type");
    mMarkerColIndex   = glGetUniformLocation(mMarkerProgram, "line_color");
    mSpriteTMatIndex  = glGetUniformLocation(mMarkerProgram, "transform");
}

surface_impl::~surface_impl()
{
    CheckGL("Begin Plot::~Plot");
    glDeleteBuffers(1, &mMainVBO);
    CheckGL("End Plot::~Plot");
}

void surface_impl::setColor(fg::Color col)
{
    mLineColor[0] = (((int) col >> 24 ) & 0xFF ) / 255.f;
    mLineColor[1] = (((int) col >> 16 ) & 0xFF ) / 255.f;
    mLineColor[2] = (((int) col >> 8  ) & 0xFF ) / 255.f;
    mLineColor[3] = (((int) col       ) & 0xFF ) / 255.f;
}

void surface_impl::setColor(float r, float g, float b)
{
    mLineColor[0] = clampTo01(r);
    mLineColor[1] = clampTo01(g);
    mLineColor[2] = clampTo01(b);
    mLineColor[3] = 1.0f;
}

GLuint surface_impl::vbo() const
{
    return mMainVBO;
}

size_t surface_impl::size() const
{
    return mMainVBOsize;
}

void surface_impl::render(int pWindowId, int pX, int pY, int pVPW, int pVPH)
{
    float range_x = xmax() - xmin();
    float range_y = ymax() - ymin();
    float range_z = zmax() - zmin();
    // set scale to zero if input is constant array
    // otherwise compute scale factor by standard equation
    float graph_scale_x = std::abs(range_x) < 1.0e-3 ? 0.0f : 2/(xmax() - xmin());
    float graph_scale_y = std::abs(range_y) < 1.0e-3 ? 0.0f : 2/(ymax() - ymin());
    float graph_scale_z = std::abs(range_z) < 1.0e-3 ? 0.0f : 2/(zmax() - zmin());

    CheckGL("Begin Plot::render");

    float coor_offset_x = ( -xmin() * graph_scale_x);
    float coor_offset_y = ( -ymin() * graph_scale_y);
    float coor_offset_z = ( -zmin() * graph_scale_z);

    glm::mat4 model = glm::rotate(glm::mat4(1.0f), -glm::radians(90.f), glm::vec3(1,0,0)) * glm::translate(glm::mat4(1.f), glm::vec3(-1 + coor_offset_x  , -1 + coor_offset_y, -1 + coor_offset_z)) *  glm::scale(glm::mat4(1.f), glm::vec3(1.0f * graph_scale_x, -1.0f * graph_scale_y, 1.0f * graph_scale_z));
    glm::mat4 view = glm::lookAt(glm::vec3(-1,0.5f,1.0f), glm::vec3(1,-1,-1),glm::vec3(0,1,0));
    glm::mat4 projection = glm::ortho(-2.f, 2.f, -2.f, 2.f, -1.1f, 100.f);
    glm::mat4 mvp = projection * view * model;
    glm::mat4 transform = mvp;
    renderGraph(pWindowId, transform);

    /* render graph border and axes */
    renderChart(pWindowId, pX, pY, pVPW, pVPH);

    CheckGL("End Plot::render");
}

void surface_impl::renderGraph(int pWindowId, glm::mat4 transform){
    bindSurfProgram();
    GLfloat range[] = {xmax(), xmin(), ymax(), ymin(), zmax(), zmin()};
    glUniform2fv(surfRangeIndex(), 3, range);
    glUniformMatrix4fv(surfMatIndex(), 1, GL_FALSE, glm::value_ptr(transform));
    glUniform4fv(borderColorIndex(), 1, mLineColor);
    bindResources(pWindowId);
    glDrawElements(GL_TRIANGLE_STRIP, mIndexVBOsize, GL_UNSIGNED_SHORT, (void*)0 );
    unbindResources();
    unbindSurfProgram();


    if(mMarkerType != fg::FG_NONE){
        glEnable(GL_PROGRAM_POINT_SIZE);
        glUseProgram(mMarkerProgram);

        glUniformMatrix4fv(spriteMatIndex(), 1, GL_FALSE, glm::value_ptr(transform));
        glUniform4fv(markerColIndex(), 1, WHITE);
        glUniform1i(markerTypeIndex(), mMarkerType);

        bindResources(pWindowId);
        glDrawElements(GL_POINTS, mIndexVBOsize, GL_UNSIGNED_SHORT, (void*)0 );
        unbindResources();
        glUseProgram(0);
        glDisable(GL_PROGRAM_POINT_SIZE);
    }
}

GLuint surface_impl::markerTypeIndex() const
{
    return mMarkerTypeIndex;
}

GLuint surface_impl::spriteMatIndex() const
{
    return mSpriteTMatIndex;
}

GLuint surface_impl::markerColIndex() const
{
    return mMarkerColIndex;
}

GLuint surface_impl::surfMatIndex() const
{
    return mSurfTMatIndex;
}

GLuint surface_impl::surfRangeIndex() const
{
    return mSurfRangeIndex;
}

void surface_impl::bindSurfProgram() const
{
    glUseProgram(mSurfProgram);
}

void surface_impl::unbindSurfProgram() const
{
    glUseProgram(0);
}


void scatter3_impl::renderGraph(int pWindowId, glm::mat4 transform){
    if(mMarkerType != fg::FG_NONE){
        glEnable(GL_PROGRAM_POINT_SIZE);
        glUseProgram(mMarkerProgram);

        glUniformMatrix4fv(spriteMatIndex(), 1, GL_FALSE, glm::value_ptr(transform));
        glUniform4fv(markerColIndex(), 1, mLineColor);
        glUniform1i(markerTypeIndex(), mMarkerType);

        bindResources(pWindowId);
        glDrawElements(GL_POINTS, mIndexVBOsize, GL_UNSIGNED_SHORT, (void*)0 );
        unbindResources();
        glUseProgram(0);
        glDisable(GL_PROGRAM_POINT_SIZE);
    }
}

}

namespace fg
{

Surface::Surface(unsigned pNumXPoints, unsigned pNumYPoints, dtype pDataType, PlotType pPlotType, MarkerType pMarkerType)
{
    value = new internal::_Surface(pNumXPoints, pNumYPoints, pDataType, pPlotType, pMarkerType);
}

Surface::Surface(const Surface& other)
{
    value = new internal::_Surface(*other.get());
}

Surface::~Surface()
{
    delete value;
}

void Surface::setColor(fg::Color col)
{
    value->setColor(col);
}

void Surface::setColor(float r, float g, float b)
{
    value->setColor(r, g, b);
}

void Surface::setAxesLimits(float pXmax, float pXmin, float pYmax, float pYmin, float pZmax, float pZmin)
{
    value->setAxesLimits(pXmax, pXmin, pYmax, pYmin, pZmax, pZmin);
}

void Surface::setXAxisTitle(const char* pTitle)
{
    value->setXAxisTitle(pTitle);
}

void Surface::setYAxisTitle(const char* pTitle)
{
    value->setYAxisTitle(pTitle);
}

void Surface::setZAxisTitle(const char* pTitle)
{
    value->setZAxisTitle(pTitle);
}


float Surface::xmax() const
{
    return value->xmax();
}

float Surface::xmin() const
{
    return value->xmin();
}

float Surface::ymax() const
{
    return value->ymax();
}

float Surface::ymin() const
{
    return value->ymin();
}

float Surface::zmax() const
{
    return value->zmax();
}

float Surface::zmin() const
{
    return value->zmin();
}

unsigned Surface::vbo() const
{
    return value->vbo();
}

unsigned Surface::size() const
{
    return (unsigned)value->size();
}

internal::_Surface* Surface::get() const
{
    return value;
}

}

/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <plot2d.hpp>
#include <common.hpp>
#include <err_common.hpp>
#include <stdexcept>
#include <iostream>
#include <cstring>
#include <cstdio>
#include <math.h>

using namespace std;
namespace backend
{
    fg_plot_handle plot_init(const fg_window_handle window, const unsigned width, const unsigned height)
    {
        fg_plot_handle plot = new fg_plot_struct[1];
        // set window here
        plot->window = window;
        plot->src_width = width;
        plot->src_height = height;
        plot->ticksize = 10;
        plot->margin = 20;
        MakeContextCurrent(plot->window);

        glGenBuffers(1, &(plot->gl_vbo));
        plot->vbosize = 0;

        GLint compile_ok = GL_FALSE, link_ok = GL_FALSE;
        // Vertex Shader
        GLuint vs = glCreateShader(GL_VERTEX_SHADER);
        const char *vs_source =
            "attribute vec2 coord2d;            "
            "varying vec4 f_color;              "
            "uniform float offset_x;            "
            "uniform float offset_y;            "
            "uniform float scale_x;             "
            "uniform float scale_y;             "
            "void main(void) {                  "
            "   gl_Position = vec4((coord2d.x + offset_x) * scale_x, (coord2d.y + offset_y) * scale_y, 0, 1);"
            "   f_color = vec4(1.0, 0, 0, 1);   "
            "   gl_PointSize = 1.0;             "
            "}";
        glShaderSource(vs, 1, &vs_source, NULL);
        glCompileShader(vs);
        glGetShaderiv(vs, GL_COMPILE_STATUS, &compile_ok);
        if ( !compile_ok)
        {
            fprintf(stderr, "Error in vertex shader\n");
        }

        // Fragment Shader
        GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
        const char *fs_source =
            "varying vec4 f_color;          "
            "void main(void) {              "
            "   gl_FragColor = f_color;     "
            "}";
        glShaderSource(fs, 1, &fs_source, NULL);
        glCompileShader(fs);
        glGetShaderiv(fs, GL_COMPILE_STATUS, &compile_ok);
        if (!compile_ok) {
            fprintf(stderr, "Error in fragment shader\n");
        }

        // Attach Shaders
        plot->gl_Program = glCreateProgram();
        glAttachShader(plot->gl_Program, vs);
        glAttachShader(plot->gl_Program, fs);
        glLinkProgram(plot->gl_Program);
        glGetProgramiv(plot->gl_Program, GL_LINK_STATUS, &link_ok);
        if (!link_ok) {
            fprintf(stderr, "Error in glLinkProgram:");
        }

        plot->gl_Attribute_Coord2d = glGetAttribLocation(plot->gl_Program, "coord2d");
        plot->gl_Uniform_Offset_x = glGetUniformLocation(plot->gl_Program, "offset_x");
        plot->gl_Uniform_Offset_y = glGetUniformLocation(plot->gl_Program, "offset_y");
        plot->gl_Uniform_Scale_x = glGetUniformLocation(plot->gl_Program, "scale_x");
        plot->gl_Uniform_Scale_y = glGetUniformLocation(plot->gl_Program, "scale_y");

        return plot;
    }

    void plot_2d(fg_plot_handle plot, const double xmax,const double xmin, const double ymax, const double ymin, const int size)
    {

        MakeContextCurrent(plot->window);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Give our program to OpenGL
        glUseProgram(plot->gl_Program);

        // Set default offset and scale variables
        float scale_x = 1/(xmax - xmin);
        float scale_y = 1/(ymax - ymin);
        float offset_x = 0;
        float offset_y = 0;


/*  TODO: The following are for the future to enable keyboard input
        if (glfwGetKey( current->pWindow, GLFW_KEY_LEFT ) == GLFW_PRESS)
            offset_x -= 0.1;
        if (glfwGetKey( current->pWindow, GLFW_KEY_RIGHT ) == GLFW_PRESS)
            offset_x += 0.1;
        if (glfwGetKey( current->pWindow, GLFW_KEY_UP ) == GLFW_PRESS)
            scale_x *= 1.5;
        if (glfwGetKey( current->pWindow, GLFW_KEY_DOWN ) == GLFW_PRESS)
            scale_x /= 1.5;
        if (glfwGetKey( current->pWindow, GLFW_KEY_HOME ) == GLFW_PRESS)
        {
            offset_x = 0.1;
            scale_x = 1.0;
        }
*/


        glUniform1f(plot->gl_Uniform_Offset_x,offset_x);
        glUniform1f(plot->gl_Uniform_Offset_y,offset_y);
        glUniform1f(plot->gl_Uniform_Scale_x,scale_x);
        glUniform1f(plot->gl_Uniform_Scale_y,scale_y);

        glBindBuffer(GL_ARRAY_BUFFER, plot->gl_vbo);

        glEnableVertexAttribArray(plot->gl_Attribute_Coord2d);
        glVertexAttribPointer(plot->gl_Attribute_Coord2d, 2, plot->window->type, GL_FALSE, 0, 0);

        glDrawArrays(GL_LINE_STRIP, 0, size);

        glfwSwapBuffers(plot->window->pWindow);
        glfwPollEvents();
    }

    void destroyPlot(fg_plot_handle plot)
    {
        CheckGL("Before destroyPlot");
        MakeContextCurrent(plot->window);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glDeleteBuffers(1, &(plot->gl_vbo));
        glDeleteProgram(plot->gl_Program);
        CheckGL("In destroyPlot");
    }
}

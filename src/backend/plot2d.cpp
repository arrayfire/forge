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

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// Struct for ticks and borders
struct point {
    GLfloat x;
    GLfloat y;
};

float offset_x = 0;
float scale_x = 1;
float offset_y = 0;
float scale_y = 1;

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
        plot->margin = 10;
        MakeContextCurrent(plot->window);

        glGenBuffers(3, (plot->gl_vbo));
        plot->vbosize = 0;

        GLint compile_ok = GL_FALSE, link_ok = GL_FALSE;

        // Vertex Shader
        GLuint vs = glCreateShader(GL_VERTEX_SHADER);
        const char *vs_source =
            "attribute vec2 coord2d;            "
            "uniform mat4 transform;            "
            "void main(void) {                  "
            "   gl_Position = transform * vec4(coord2d.xy, 0, 1);"
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
            "uniform vec4 color;          "
            "void main(void) {            "
            "   gl_FragColor = color;     "
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
        plot->gl_Uniform_Color     = glGetUniformLocation(plot->gl_Program, "color");
        plot->gl_Uniform_Transform     = glGetUniformLocation(plot->gl_Program, "transform");

        // VBO for border
        static const point border[4] = { {-1, -1}, {1, -1}, {1, 1}, {-1, 1} };
        glBindBuffer(GL_ARRAY_BUFFER, plot->gl_vbo[1]);
        glBufferData(GL_ARRAY_BUFFER, sizeof (border), border, GL_STATIC_DRAW);

        return plot;
    }

    glm::mat4 viewport_transform(fg_plot_handle plot, float x, float y, float width, float height, float *pixel_x = 0, float *pixel_y = 0)
    {
        float window_width  = plot->src_width;
        float window_height = plot->src_height;

        float offset_x = (2.0 * x + (width - window_width))   / window_width;
        float offset_y = (2.0 * y + (height - window_height)) / window_height;

        float scale_x = width  / window_width;
        float scale_y = height / window_height;

        if (pixel_x)
            *pixel_x = 2.0 / width;
        if (pixel_y)
            *pixel_y = 2.0 / height;

        return glm::scale(glm::translate(glm::mat4(1), glm::vec3(offset_x, offset_y, 0)), glm::vec3(scale_x, scale_y, 1));
    }

    void plot_2d(fg_plot_handle plot, const double xmax,const double xmin, const double ymax, const double ymin)
    {
        ForceCheckGL("Before plot_2d");
        MakeContextCurrent(plot->window);
        glClearColor(1, 1, 1, 1);
        glClear(GL_COLOR_BUFFER_BIT);

        // Give our program to OpenGL
        glUseProgram(plot->gl_Program);

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
        // Draw graph

        // Set viewport. This will clip geometry
        glViewport(plot->margin + plot->ticksize, plot->margin + plot->ticksize, plot->src_width - plot->margin * 2 - plot->ticksize, plot->src_height - plot->margin * 2 - plot->ticksize);

        // Set scissor rectangle to clip fragments outside of viewport
        glScissor(plot->margin + plot->ticksize, plot->margin + plot->ticksize, plot->src_width - plot->margin * 2 - plot->ticksize, plot->src_height - plot->margin * 2 - plot->ticksize);

        glEnable(GL_SCISSOR_TEST);

        float graph_scale_x = 1/(xmax - xmin);
        float graph_scale_y = 1/(ymax - ymin);

        glm::mat4 transform = glm::translate(glm::scale(glm::mat4(1.0f), glm::vec3(graph_scale_x, graph_scale_y, 1)), glm::vec3(offset_x, 0, 0));
        glUniformMatrix4fv(plot->gl_Uniform_Transform, 1, GL_FALSE, glm::value_ptr(transform));

        // Set the color to red
        GLfloat red[4] = { 1, 0, 0, 1 };
        glUniform4fv(plot->gl_Uniform_Color, 1, red);
        glBindBuffer(GL_ARRAY_BUFFER, plot->gl_vbo[0]);
        ForceCheckGL("Before enable vertex attribute array");

        glEnableVertexAttribArray(plot->gl_Attribute_Coord2d);
        glVertexAttribPointer(plot->gl_Attribute_Coord2d, 2, plot->window->type, GL_FALSE, 0, 0);
        ForceCheckGL("Before setting elements");

        size_t elements = 0;
        switch(plot->window->type) {
            case GL_FLOAT:          elements = plot->vbosize / (2 * sizeof(float));     break;
            case GL_INT:            elements = plot->vbosize / (2 * sizeof(int  ));     break;
            case GL_UNSIGNED_INT:   elements = plot->vbosize / (2 * sizeof(uint ));     break;
            case GL_BYTE:           elements = plot->vbosize / (2 * sizeof(char ));     break;
            case GL_UNSIGNED_BYTE:  elements = plot->vbosize / (2 * sizeof(uchar));     break;
        }
        glDrawArrays(GL_LINE_STRIP, 0, elements);

        // Stop clipping
        glViewport(0, 0, plot->src_width, plot->src_height);
        glDisable(GL_SCISSOR_TEST);

        // Draw borders
        float pixel_x, pixel_y;
        float margin        = plot->margin;
        float ticksize      = plot->ticksize;
        float window_width  = plot->src_width;
        float window_height = plot->src_height;

        transform = viewport_transform(plot, margin + ticksize, margin + ticksize, window_width - 2 * margin - ticksize, window_height - 2 * margin - ticksize, &pixel_x, &pixel_y);
        glUniformMatrix4fv(plot->gl_Uniform_Transform, 1, GL_FALSE, glm::value_ptr(transform));

        // Set the color to black
        GLfloat black[4] = { 0, 0, 0, 1 };
        glUniform4fv(plot->gl_Uniform_Color, 1, black);

        glBindBuffer(GL_ARRAY_BUFFER, plot->gl_vbo[1]);
        glVertexAttribPointer(plot->gl_Attribute_Coord2d, 2, plot->window->type, GL_FALSE, 0, 0);

        glDrawArrays(GL_LINE_LOOP, 0, 4);

        // Draw y tick marks
        point ticks[42];
        float ytickspacing = 0.1 * powf(10, -floor(log10(scale_y)));
        float top = -1.0 / scale_y - offset_y;       // top edge, in graph coordinates
        float bottom = 1.0 / scale_y - offset_y;     // right edge, in graph coordinates
        int top_i = ceil(top / ytickspacing);        // index of top tick, counted from the origin
        int bottom_i = floor(bottom / ytickspacing); // index of bottom tick, counted from the origin
        float y_rem = top_i * ytickspacing - top;    // space between top edge of graph and the first tick

        float y_firsttick = -1.0 + y_rem * scale_y;  // first tick in device coordinates

        int y_nticks = bottom_i - top_i + 1;         // number of y ticks to show
        if (y_nticks > 21)
            y_nticks = 21;    // should not happen

        for (int i = 0; i <= y_nticks; i++) {
            float y = y_firsttick + i * ytickspacing * scale_y;
            float ytickscale = ((i + top_i) % 10) ? 0.5 : 1;

            ticks[i * 2].x = -1;
            ticks[i * 2].y = y;
            ticks[i * 2 + 1].x = -1 - ticksize * ytickscale * pixel_x;
            ticks[i * 2 + 1].y = y;
        }

        glBindBuffer(GL_ARRAY_BUFFER, plot->gl_vbo[2]);
        glBufferData(GL_ARRAY_BUFFER, sizeof(ticks), ticks, GL_DYNAMIC_DRAW);
        glVertexAttribPointer(plot->gl_Attribute_Coord2d, 2, GL_FLOAT, GL_FALSE, 0, 0);
        glDrawArrays(GL_LINES, 0, y_nticks * 2);

        // Draw x tick marks
        ForceCheckGL("Before x ticks");
        float xtickspacing = 0.1 * powf(10, -floor(log10(scale_x)));
        float left = -1.0 / scale_x - offset_x;     // left edge, in graph coordinates
        float right = 1.0 / scale_x - offset_x;     // right edge, in graph coordinates
        int left_i = ceil(left / xtickspacing);     // index of left tick, counted from the origin
        int right_i = floor(right / xtickspacing);  // index of right tick, counted from the origin
        float x_rem = left_i * xtickspacing - left; // space between left edge of graph and the first tick

        float x_firsttick = -1.0 + x_rem * scale_x; // first tick in device coordinates

        int x_nticks = right_i - left_i + 1;        // number of x ticks to show

        if (x_nticks > 21)
            x_nticks = 21;    // should not happen

        for (int i = 0; i < x_nticks; i++) {
            float x = x_firsttick + i * xtickspacing * scale_x;
            float xtickscale = ((i + left_i) % 10) ? 0.5 : 1;

            ticks[i * 2].x = x;
            ticks[i * 2].y = -1;
            ticks[i * 2 + 1].x = x;
            ticks[i * 2 + 1].y = -1 - ticksize * xtickscale * pixel_y;
        }

        glBufferData(GL_ARRAY_BUFFER, sizeof(ticks), ticks, GL_DYNAMIC_DRAW);
        glVertexAttribPointer(plot->gl_Attribute_Coord2d, 2, GL_FLOAT, GL_FALSE, 0, 0);
        glDrawArrays(GL_LINES, 0, x_nticks * 2);

        glfwSwapBuffers(plot->window->pWindow);
        glfwPollEvents();
    }

    void destroyPlot(fg_plot_handle plot)
    {
        CheckGL("Before destroyPlot");
        MakeContextCurrent(plot->window);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glDeleteBuffers(3, (plot->gl_vbo));
        glDeleteProgram(plot->gl_Program);
        CheckGL("In destroyPlot");

    }
}

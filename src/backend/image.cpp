/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

// Parts of this code sourced from SnopyDogy
// https://gist.github.com/SnopyDogy/a9a22497a893ec86aa3e

#include <image.hpp>
#include <common.hpp>
#include <err_common.hpp>

#include <stdexcept>
#include <iostream>
#include <cstring>
#include <cstdio>

namespace backend
{
    static const char* vertex_shader_code = "#version 330\n"
                                            "layout(location = 0) in vec3 pos;\n"
                                            "layout(location = 1) in vec2 tex;\n"
                                            "uniform mat4 matrix;\n"
                                            "out vec2 texcoord;\n"
                                            "void main() {\n"
                                            "    texcoord = tex;\n"
                                            "    gl_Position = matrix * vec4(pos,1.0);\n"
                                            "}\n";

    static const char* fragment_shader_code = "#version 330\n"
                                              "uniform sampler2D tex;\n"
                                              "in vec2 texcoord;\n"
                                              "out vec4 fragColor;\n"
                                              "void main()\n"
                                              "{\n"
                                              "    fragColor = texture2D(tex,texcoord);\n"
                                              "}\n";

    static GLuint gCanvasVAO = 0;

    template<typename T>
    fg_image_handle setupImage(fg_window_handle window, const unsigned width, const unsigned height)
    {
        CheckGL("Before setupImage");
        fg_image_handle image = new fg_image_struct[1];
        image->window = window;
        MakeContextCurrent(image->window);

        image->gl_Format = mode_to_glColor(image->window->mode);
        image->gl_Type   = window->type;

        image->src_width = width;
        image->src_height = height;

        CheckGL("Before Texture Initialization");
        // Initialize OpenGL Items
        glEnable(GL_TEXTURE_2D);
        glGenTextures(1, &(image->gl_Tex));
        glBindTexture(GL_TEXTURE_2D, image->gl_Tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

        glTexImage2D(GL_TEXTURE_2D, 0, image->gl_Format, image->src_width, image->src_height, 0, image->gl_Format, image->gl_Type, NULL);

        CheckGL("Before PBO Initialization");
        glGenBuffers(1, &(image->gl_PBO));
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, image->gl_PBO);
        glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, image->src_width * image->src_height * image->window->mode * sizeof(T), NULL, GL_STREAM_COPY);

        CheckGL("Before Shader Initialization");
        // load shader program
        image->gl_Shader = initShaders(vertex_shader_code, fragment_shader_code);

        if (gCanvasVAO==0) {
            const float vertices[12] = {
                -1.0f,-1.0f,0.0,
                1.0f,-1.0f,0.0,
                1.0f, 1.0f,0.0,
                -1.0f, 1.0f,0.0};

            const float texcords[8] = {0.0,1.0,1.0,1.0,1.0,0.0,0.0,0.0};

            const uint indices[6] = {0,1,2,0,2,3};

            GLuint vbo  = createBuffer(12, vertices, GL_STATIC_DRAW);
            GLuint tbo  = createBuffer(8, texcords, GL_STATIC_DRAW);
            GLuint ibo;
            glGenBuffers(1,&ibo);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(uint)*6, indices, GL_STATIC_DRAW);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,0);
            // bind vao
            glGenVertexArrays(1, &gCanvasVAO);
            glBindVertexArray(gCanvasVAO);
            // attach vbo
            glBindBuffer(GL_ARRAY_BUFFER, vbo);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
            glEnableVertexAttribArray(0);
            // attach tbo
            glBindBuffer(GL_ARRAY_BUFFER,tbo);
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, NULL);
            glEnableVertexAttribArray(1);
            // attach ibo
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
            glBindVertexArray(0);
        }

        glBindTexture(GL_TEXTURE_2D, 0);

        CheckGL("At End of setupImage");
        return image;
    }

#define INSTANTIATE(T)                                                                                      \
    template fg_image_handle setupImage<T>(fg_window_handle window, const unsigned width, const unsigned height);   \

    INSTANTIATE(float);
    INSTANTIATE(int);
    INSTANTIATE(unsigned);
    INSTANTIATE(char);
    INSTANTIATE(unsigned char);

#undef INSTANTIATE

    void renderImage(const fg_image_handle& image)
    {
        static const float matrix[16] = {
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f};

        glUseProgram(image->gl_Shader);
        // get uniform locations
        int mat_loc = glGetUniformLocation(image->gl_Shader,"matrix");
        int tex_loc = glGetUniformLocation(image->gl_Shader,"tex");

        // load texture from PBO
        glActiveTexture(GL_TEXTURE0);
        glUniform1i(tex_loc, 0);
        glBindTexture(GL_TEXTURE_2D, image->gl_Tex);
        // bind PBO to load data into texture
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, image->gl_PBO);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, image->src_width, image->src_height, image->gl_Format, image->gl_Type, 0);

        glUniformMatrix4fv(mat_loc, 1, GL_FALSE, matrix);

        // Draw to screen
        glBindVertexArray(gCanvasVAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);

        // Unbind textures
        glBindTexture(GL_TEXTURE_2D, 0);

        // ubind the shader program
        glUseProgram(0);
    }

    void drawImage(const fg_image_handle image)
    {
        MakeContextCurrent(image->window);

        int wind_width, wind_height;
        glfwGetWindowSize(image->window->pWindow, &wind_width, &wind_height);
        glViewport(0, 0, wind_width, wind_height);

        CheckGL("Before drawImage");
        // clear color and depth buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(0.2, 0.2, 0.2, 1.0);

        renderImage(image);

        glfwSwapBuffers(image->window->pWindow);
        glfwPollEvents();
        ForceCheckGL("In drawImage");
    }

    void drawImages(int pRows, int pCols, const unsigned int pNumImages, const fg_image_handle pHandles[])
    {
        MakeContextCurrent(pHandles[0]->window);

        int wind_width, wind_height;
        glfwGetWindowSize(pHandles[0]->window->pWindow, &wind_width, &wind_height);
        glViewport(0, 0, wind_width, wind_height);

        // calculate cell width and height
        uint wid_step = wind_width/ pCols;
        uint hei_step = wind_height/ pRows;

        CheckGL("Before drawImages");
        // clear color and depth buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(0.2, 0.2, 0.2, 1.0);

        for (int c=0; c<pCols; ++c) {
            for (int r=0; r<pRows; ++r) {
                uint idx = c*pRows + pRows-1-r;
                if (idx<pNumImages) {

                    int x_off = c * wid_step;
                    int y_off = r * hei_step;

                    // set viewport to render sub image
                    glViewport(x_off, y_off, wid_step, hei_step);

                    renderImage(pHandles[idx]);
                }
            }
        }

        glfwSwapBuffers(pHandles[0]->window->pWindow);
        glfwPollEvents();
        ForceCheckGL("In drawImages");
    }

    void destroyImage(const fg_image_handle image)
    {
        CheckGL("Before destroyImage");
        // Cleanup
        MakeContextCurrent(image->window);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
        glDeleteBuffers(1, &image->gl_PBO);
        glDeleteTextures(1, &image->gl_Tex);
        glDeleteProgram(image->gl_Shader);

        CheckGL("In destoryImage");
    }
}

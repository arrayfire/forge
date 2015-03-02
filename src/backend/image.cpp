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
    // gl_Shader for displaying floating-point texture
    static const char *shader_code =
        "!!ARBfp1.0\n"
        "TEX result.color, fragment.texcoord, texture[0], 2D; \n"
        "END";

    GLuint compileASMShader(GLenum program_type, const char *code)
    {
        GLuint program_id;
        glGenProgramsARB(1, &program_id);
        glBindProgramARB(program_type, program_id);
        glProgramStringARB(program_type, GL_PROGRAM_FORMAT_ASCII_ARB, (GLsizei) strlen(code), (GLubyte *) code);

        GLint error_pos;
        glGetIntegerv(GL_PROGRAM_ERROR_POSITION_ARB, &error_pos);

        if (error_pos != -1)
        {
            const GLubyte *error_string;
            error_string = glGetString(GL_PROGRAM_ERROR_STRING_ARB);
            fprintf(stderr, "Program error at position: %d\n%s\n", (int)error_pos, error_string);
            return 0;
        }

        return program_id;
    }

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
        image->gl_Shader = compileASMShader(GL_FRAGMENT_PROGRAM_ARB, shader_code);

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

    void drawImage(const fg_image_handle image)
    {
        CheckGL("Before drawImage");
        // Cleanup
        MakeContextCurrent(image->window);

        // load texture from PBO
        glBindTexture(GL_TEXTURE_2D, image->gl_Tex);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, image->src_width, image->src_height, image->gl_Format, image->gl_Type, 0);

        // fragment program is required to display floating point texture
        glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, image->gl_Shader);
        glEnable(GL_FRAGMENT_PROGRAM_ARB);
        glDisable(GL_DEPTH_TEST);

        // Draw to screen
        // GLFW uses -1 to 1 normalized coordinates
        // Textures go from 0 to 1 normalized coordinates
        glBegin(GL_QUADS);
        glTexCoord2f ( 0.0f,  1.0f);
        glVertex2f   (-1.0f, -1.0f);
        glTexCoord2f ( 1.0f,  1.0f);
        glVertex2f   ( 1.0f, -1.0f);
        glTexCoord2f ( 1.0f,  0.0f);
        glVertex2f   ( 1.0f,  1.0f);
        glTexCoord2f ( 0.0f,  0.0f);
        glVertex2f   (-1.0f,  1.0f);
        glEnd();

        // Unbind textures
        glBindTexture(GL_TEXTURE_2D, 0);
        glDisable(GL_FRAGMENT_PROGRAM_ARB);

        // Complete render
        glfwSwapBuffers(current->pWindow);
        glfwPollEvents();

        ForceCheckGL("In drawImage");
    }

    void destroyImage(const fg_image_handle image)
    {
        CheckGL("Before destroyImage");
        // Cleanup
        MakeContextCurrent(image->window);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
        glDeleteBuffers(1, &image->gl_PBO);
        glDeleteTextures(1, &image->gl_Tex);
        glDeleteProgramsARB(1, &image->gl_Shader);

        CheckGL("In destoryImage");
    }
}

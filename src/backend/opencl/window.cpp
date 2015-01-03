/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <window.hpp>
#include <err_opencl.hpp>

#include <stdexcept>
#include <iostream>
#include <cstring>
#include <cstdio>

namespace opencl
{
    static int g_uiWindowCounter = 0; // Window Counter
    WindowHandle current;

    // Required to be defined for GLEW MX to work, along with the GLEW_MX define in the perprocessor!
    static GLEWContext* glewGetContext()
    {
        return current->pGLEWContext;
    }

    static void error_callback(int error, const char* description)
    {
        fputs(description, stderr);
        FW_ERROR("Error in GLFW", FW_ERR_GL_ERROR);
    }

    static void key_callback(GLFWwindow* wind, int key, int scancode, int action, int mods)
    {
        if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        {
            glfwSetWindowShouldClose(wind, GL_TRUE);
        }
    }

    void MakeContextCurrent(WindowHandle wh)
    {
        CheckGL("Before MakeContextCurrent");
        if (wh != NULL)
        {
            glfwMakeContextCurrent(wh->pWindow);
            current = wh;
        }
        CheckGL("In MakeContextCurrent");
    }

    template<typename T>
    WindowHandle createWindow(const char *title, const unsigned disp_w, const unsigned disp_h,
                              const fw_color_mode mode)
    {
        // save current active context info so we can restore it later!
        //WindowHandle previous = current;

        // create new window data:
        WindowHandle newWindow = new fw_window;
        if (newWindow == NULL)
            printf("Error\n");
            //Error out

        newWindow->pGLEWContext = NULL;
        newWindow->pWindow      = NULL;
        newWindow->uiID         = g_uiWindowCounter++;        //set ID and Increment Counter!
        newWindow->uiWidth      = disp_w;
        newWindow->uiHeight     = disp_h;

        // Initalize GLFW
        glfwSetErrorCallback(error_callback);
        if (!glfwInit()) {
            std::cerr << "ERROR: GLFW wasn't able to initalize" << std::endl;
            exit(EXIT_FAILURE);
        }

        // Add Hints
        glfwWindowHint(GLFW_DEPTH_BITS, mode * sizeof(T));
        glfwWindowHint(GLFW_RESIZABLE, false);

        // Create the window itself
        newWindow->pWindow = glfwCreateWindow(newWindow->uiWidth, newWindow->uiHeight, title, NULL, NULL);

        // Confirm window was created successfully:
        if (newWindow->pWindow == NULL)
        {
            printf("Error: Could not Create GLFW Window!\n");
            delete newWindow;
            return NULL;
        }

        // Create GLEW Context
        newWindow->pGLEWContext = new GLEWContext();
        if (newWindow->pGLEWContext == NULL)
        {
            printf("Error: Could not create GLEW Context!\n");
            delete newWindow;
            return NULL;
        }

        // Set context (before glewInit())
        MakeContextCurrent(newWindow);

        //GLEW Initialization - Must be done
        GLenum err = glewInit();
        if (err != GLEW_OK) {
            printf("GLEW Error occured, Description: %s\n", glewGetErrorString(err));
            glfwDestroyWindow(newWindow->pWindow);
            delete newWindow;
            return NULL;
        }

        int b_width  = newWindow->uiWidth;
        int b_height = newWindow->uiHeight;
        glfwGetFramebufferSize(newWindow->pWindow, &b_width, &b_height);

        glViewport(0, 0, b_width, b_height);

        glfwSetKeyCallback(newWindow->pWindow, key_callback);

        MakeContextCurrent(newWindow);

        CheckGL("At End of Create Window");
        return newWindow;
    }

    void destroyWindow(WindowHandle window)
    {
        CheckGL("Before Delete Window");
        // Cleanup
        MakeContextCurrent(window);

        // Delete GLEW context and GLFW window
        delete window->pGLEWContext;
        glfwDestroyWindow(window->pWindow);

        CheckGL("In Delete Window");
    }
}

include(hunter_config)

macro(myhunter_config pkg_name ver)
    hunter_config(
        ${pkg_name}
        VERSION ${ver}
        CONFIGURATION_TYPES Release
        CMAKE_ARGS CMAKE_POSITION_INDEPENDENT_CODE=ON ${ARGN}
    )
    mark_as_advanced(${pkg_name}_DIR)
endmacro()

myhunter_config(Boost 1.66.0)
myhunter_config(freetype 2.6.2)
myhunter_config(OpenCL 2.1-p0)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "7.2.0")
        myhunter_config(glm 0.9.7.6)
    else()
        myhunter_config(glm 0.9.8.5)
    endif()
endif ()

#myhunter_config(freeimage hunter-v3.17.0)
##freeimag dependencies
#myhunter_config(ZLIB 1.2.8-p3)
#myhunter_config(TIFF 4.0.2-p3)
#myhunter_config(PNG 1.6.26-p1)
#myhunter_config(Jpeg 9b-p3)

#glfw dependencies
myhunter_config(xcursor 1.1.13)
myhunter_config(xorg-macros 1.17)
myhunter_config(xrender 0.9.7)
myhunter_config(x11 1.5.0)
myhunter_config(xproto 7.0.23)
myhunter_config(xextproto 7.2.1)
myhunter_config(xtrans 1.2.7)
myhunter_config(xcb 1.11.1)
myhunter_config(xcb-proto 1.11)
myhunter_config(pthread-stubs 0.3)
myhunter_config(xau 1.0.7)
myhunter_config(kbproto 1.0.6)
myhunter_config(inputproto 2.2)
myhunter_config(renderproto 0.11.1)
myhunter_config(xfixes 5.0.1)
myhunter_config(fixesproto 5.0)
myhunter_config(xinerama 1.1.2)
myhunter_config(xineramaproto 1.1.2)
myhunter_config(xrandr 1.3.2)
myhunter_config(randrproto 1.3.2)
myhunter_config(xi 1.6.1)
myhunter_config(xext 1.3.1)

myhunter_config(glbinding 2.1.3-p0
    OPTION_BUILD_GPU_TESTS=ON
    OPTION_BUILD_TESTS=ON
    OPTION_BUILD_TOOLS=ON
    gtest_force_shared_crt=ON)

myhunter_config(glfw 3.3.0-p4
    GLFW_BUILD_DOCS=OFF
    GLFW_BUILD_EXAMPLES=OFF
    GLFW_BUILD_TESTS=OFF)

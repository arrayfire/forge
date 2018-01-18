find_package(Qt5Widgets REQUIRED)
find_package(Qt5Gui REQUIRED)
find_package(Qt5OpenGL REQUIRED)

add_library(wtk_interface INTERFACE)

#qt5_wrap_cpp(WTK_MOC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/dockWrapper.hpp)
#set_source_files_properties(${WTK_MOC_FILES} PROPERTIES GENERATED TRUE) 
#set_target_properties(${BackendTarget} PROPERTIES AUTOMOC TRUE)

target_include_directories(wtk_interface
    INTERFACE
        ${CMAKE_SOURCE_DIR}/src/backend/opengl
    )

target_sources(wtk_interface
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/dockWrapper.hpp
#        ${WTK_MOC_FILES}
        ${CMAKE_CURRENT_SOURCE_DIR}/dockWrapper.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/window.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/window.cpp
    )

target_link_libraries(wtk_interface INTERFACE Qt5::Core Qt5::Widgets Qt5::Gui Qt5::OpenGL)

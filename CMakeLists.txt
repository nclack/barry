cmake_minimum_required(VERSION 2.8)
project(mltk-barry)

include(cmake/show.cmake)
include(cmake/ide.cmake)
include(cmake/rpath.cmake)

file(GLOB_RECURSE SRCS src/*.c src/*.h)
add_library(libengine STATIC ${SRCS})
INCLUDE_DIRECTORIES(src)

set(ENGINE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
function(target_add_engine tgt)
    set_property(TARGET ${tgt} APPEND PROPERTY 
                 INCLUDE_DIRECTORIES ${ENGINE_DIR}) 
    target_link_libraries(${tgt} libengine)
endfunction()

include(CTest)
enable_testing()
add_subdirectory(test)

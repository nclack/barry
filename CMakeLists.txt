cmake_minimum_required(VERSION 2.8)
project(mltk-barry)

include(cmake/show.cmake)
include(cmake/ide.cmake)
include(cmake/rpath.cmake)

find_package(ND)
find_package(CUDA 5.0)

file(GLOB_RECURSE SRCS src/*.c src/*.h)
include_directories(src)

file(GLOB_RECURSE CUSRCS src/*.cu)
cuda_compile(CUFILES ${CUSRCS} STATIC
      OPTIONS
      DEBUG -g;-G;-lineinfo
      RELEASE --use_fast_math
      RELWITHDEBINFO --use_fast_math;-g;-G;-lineinfo
      MINSIZEREL --use_fast_math
)
list(APPEND SRCS ${CUFILES})

add_library(libengine STATIC ${SRCS})

set(ENGINE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
function(target_add_engine tgt)
    set_property(TARGET ${tgt} APPEND PROPERTY 
                 INCLUDE_DIRECTORIES ${ENGINE_DIR})
    target_link_libraries(${tgt} libengine)
    nd_add_to_target(${tgt} ndio-tiff ndio-ffmpeg)
endfunction()

include(CTest)
enable_testing()
add_subdirectory(test)

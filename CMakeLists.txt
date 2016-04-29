cmake_minimum_required(VERSION 3.0)
project(barry)

include(cmake/show.cmake)
include(cmake/ide.cmake)
include(cmake/rpath.cmake)

find_package(CUDA 5.0)

file(GLOB_RECURSE SRCS src/*.c src/*.h)
include_directories(src)

file(GLOB_RECURSE CUSRCS src/*.cu)
cuda_compile(CUFILES ${CUSRCS} STATIC
      OPTIONS
      DEBUG -g;-G;-lineinfo;-Xptxas="-v"
      RELEASE --use_fast_math
      RELWITHDEBINFO --use_fast_math;-g;-G;-lineinfo
      MINSIZEREL --use_fast_math
)
list(APPEND SRCS ${CUFILES})

add_library(barry STATIC ${SRCS})

set(BARRY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
function(target_add_barry tgt)
  target_include_directories(${tgt} PRIVATE ${BARRY_DIR})
  target_link_libraries(${tgt} barry ${CUDA_LIBRARIES})
endfunction()

include(CTest)
enable_testing()
add_subdirectory(test)

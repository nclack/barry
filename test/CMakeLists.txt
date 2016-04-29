set(prefix barry)

file(GLOB TEST_COMMON_SRCS common/*.h common/*.c)

foreach(testset
        math
        barycentricCPU
        barycentricGPU
    )
    file(GLOB TESTS ${testset}/*.c)
    foreach(t ${TESTS})
      get_filename_component(exe ${t} NAME_WE)
      set(tgt "${prefix}-${testset}-${exe}-test")
      set(tst "${prefix}-${testset}-${exe}")
      add_executable(${tgt} ${t} ${TEST_COMMON_SRCS})
      target_include_directories(${tgt}
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/common)
      set_property(
        TARGET ${tgt}
        PROPERTY FOLDER engine/test/${testset})
      target_add_barry(${tgt})
      add_test(${tst} ${tgt})
    endforeach()
endforeach()

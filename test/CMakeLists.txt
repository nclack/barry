set(tgt mltk)

foreach(testset math)
    file(GLOB TESTS ${testset}/*.c)
    foreach(t ${TESTS})
      get_filename_component(exe ${t} NAME_WE)
      add_executable("${tgt}-${testset}-${exe}-test" ${t} ${SRC})
      set_target_properties("${tgt}-${testset}-${exe}-test" PROPERTIES FOLDER engine/test/${testset})
      add_test("${tgt}-${testset}-${exe}" "${tgt}-${testset}-${exe}-test")
      target_add_engine("${tgt}-${testset}-${exe}-test")
    endforeach()
endforeach()


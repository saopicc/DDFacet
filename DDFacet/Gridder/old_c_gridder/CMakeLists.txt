#Ensure __init__.py is added to out-of-source build directory
execute_process(COMMAND touch __init__.py
		WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

#compile and link _pyGridder.so
add_library(_pyGridder Gridder.c Gridder.h)
set_target_properties(_pyGridder PROPERTIES PREFIX "") #remove "lib" prefix from library (PEP8 compliance)
target_link_libraries(_pyGridder ${RT_LIBRARIES} ${PYTHON_LIBRARY})

add_custom_command(
        TARGET _pyGridder POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_CURRENT_BINARY_DIR}/_pyGridder.so
                ${CMAKE_CURRENT_BINARY_DIR}/../_pyGridder.so)

if (${COMPILE_OLD_GRIDDER})
  #compile and link _pyGridderSmearPols.so
  add_library(_pyGridderSmearPols GridderSmearPols.c GridderSmearPols.h Semaphores.h)
  set_target_properties(_pyGridderSmearPols PROPERTIES PREFIX "") #remove "lib" prefix from library (PEP8 compliance)
  target_link_libraries(_pyGridderSmearPols ${RT_LIBRARIES} ${PYTHON_LIBRARY})

  add_custom_command(
        TARGET _pyGridderSmearPols POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_CURRENT_BINARY_DIR}/_pyGridderSmearPols.so
                ${CMAKE_CURRENT_BINARY_DIR}/../_pyGridderSmearPols.so)
endif (${COMPILE_OLD_GRIDDER})
#Ensure __init__.py is added to out-of-source build directory
execute_process(COMMAND touch __init__.py
		WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

#compile and link _pyGridder.so
add_library(_pyGridder${PYVER} Gridder.c Gridder.h)
set_target_properties(_pyGridder${PYVER} PROPERTIES PREFIX "") #remove "lib" prefix from library (PEP8 compliance)
target_link_libraries(_pyGridder${PYVER} ${RT_LIBRARIES} ${PYTHON_LIBRARY})

add_custom_command(
        TARGET _pyGridder${PYVER} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_CURRENT_BINARY_DIR}/_pyGridder${PYVER}.so
                ${CMAKE_CURRENT_BINARY_DIR}/../_pyGridder${PYVER}.so)

install(TARGETS _pyGridder${PYVER} LIBRARY DESTINATION DDFacet/cbuild/Gridder)

#compile and link _pyGridderSmearPols.so
add_library(_pyGridderSmearPolsClassic${PYVER} GridderSmearPols.c GridderSmearPols.h Semaphores.h)
set_target_properties(_pyGridderSmearPolsClassic${PYVER} PROPERTIES PREFIX "") #remove "lib" prefix from library (PEP8 compliance)
target_link_libraries(_pyGridderSmearPolsClassic${PYVER} ${RT_LIBRARIES} ${PYTHON_LIBRARY})

add_custom_command(
      TARGET _pyGridderSmearPolsClassic${PYVER} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy
	      ${CMAKE_CURRENT_BINARY_DIR}/_pyGridderSmearPolsClassic${PYVER}.so
	      ${CMAKE_CURRENT_BINARY_DIR}/../_pyGridderSmearPolsClassic${PYVER}.so)

install(TARGETS _pyGridderSmearPolsClassic${PYVER} LIBRARY DESTINATION DDFacet/cbuild/Gridder)
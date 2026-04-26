cmake_minimum_required(VERSION 3.16...3.27)
project(xrt_sorter)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Geant4 REQUIRED ui_all vis_all)
include(${Geant4_USE_FILE})

file(GLOB sources ${PROJECT_SOURCE_DIR}/src/*.cc)
file(GLOB headers ${PROJECT_SOURCE_DIR}/include/*.hh)

add_executable(xrt_sorter exampleB1.cc ${sources} ${headers})
target_include_directories(xrt_sorter PRIVATE include)
target_link_libraries(xrt_sorter PRIVATE ${Geant4_LIBRARIES})

set(XRT_SORTER_SCRIPTS
  init_vis.mac
  run1.mac
  run2.mac
  vis.mac
)

foreach(_script ${XRT_SORTER_SCRIPTS})
  configure_file(
    ${PROJECT_SOURCE_DIR}/${_script}
    ${PROJECT_BINARY_DIR}/${_script}
    COPYONLY
  )
endforeach()

install(TARGETS xrt_sorter DESTINATION bin)

cmake_minimum_required(VERSION 3.3)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
project(slam)
set(ENTRY_FILE "main.cpp")

file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")
set(INCLUDE_DIRS "./include")

add_executable(slam ${PROJECT_SOURCE_DIR}/src/${ENTRY_FILE})

if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
    message("\nbuilding for aarch64")
    #opencv
    find_package( OpenCV 4.0.0 REQUIRED )
    find_package(CUDA  REQUIRED)
    find_package(jetson-utils REQUIRED)
    include_directories("/usr/local/include/jetson-utils")
    include_directories("${CUDA_INCLUDE_DIRS}")
    target_include_directories(slam PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_include_directories(slam PRIVATE ${INCLUDE_DIRS})
    target_link_libraries(slam ${OpenCV_LIBS})
    target_link_libraries(slam jetson-utils)
endif()


# find_package(cnpy REQUIRED)
# target_link_libraries(slam "/usr/local/cnpy.h")
# target_link_libraries(slam "/home/nicklaspank/other_builds/cnpy/cnpy/cnpy.cpp")
# target_link_libraries(slam cnpy)
# message("\n\n")
# message(${cnpy_INCLUDE_DIRS})
# message("\n\n")

find_package(ZLIB REQUIRED)

include_directories(${ZLIB_INCLUDE_DIRS})

add_library(cnpy SHARED "/home/nicklaspank/other_builds/cnpy/cnpy/cnpy.cpp")
target_link_libraries(cnpy ${ZLIB_LIBRARIES})
target_link_libraries(slam cnpy)
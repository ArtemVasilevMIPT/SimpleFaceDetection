cmake_minimum_required(VERSION 3.17)

set(CMAKE_CXX_STANDARD 17)

project(OpenCVFaceDet)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../bin)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(OpenCVFaceDet main.cpp)
target_link_libraries(OpenCVFaceDet ${OpenCV_LIBS})
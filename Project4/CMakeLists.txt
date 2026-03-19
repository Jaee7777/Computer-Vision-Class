cmake_minimum_required(VERSION 3.10)
project(vidDisplay)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(vidDisplay vidDisplay.cpp)

target_link_libraries(vidDisplay ${OpenCV_LIBS})
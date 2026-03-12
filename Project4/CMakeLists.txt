cmake_minimum_required(VERSION 3.10)

project(CornerDetection)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(vidDisplay vidDisplay.cpp)

target_link_libraries(vidDisplay ${OpenCV_LIBS})
cmake_minimum_required(VERSION 3.10)
project( CS5330_Project2 )

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_SOURCE_DIR}/include)

# OpenCV
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# ONNX Runtime
set(ONNXRUNTIME_ROOT "$ENV{HOME}/libs/onnxruntime-linux-x64-1.17.1")
include_directories(${ONNXRUNTIME_ROOT}/include)
link_directories(${ONNXRUNTIME_ROOT}/lib)

# Executable
add_executable( readfiles src/readfiles.cpp src/faceDetect.cpp )
target_link_libraries( readfiles ${OpenCV_LIBS} onnxruntime )

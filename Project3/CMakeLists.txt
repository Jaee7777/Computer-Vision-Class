cmake_minimum_required(VERSION 3.16)
project( CS5330_Project1 )

set(CMAKE_CXX_STANDARD 17)

# OpenCV
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# ONNX Runtime
set(ONNXRUNTIME_ROOT "$ENV{HOME}/libs/onnxruntime-linux-x64-1.17.1")
include_directories(${ONNXRUNTIME_ROOT}/include)
link_directories(${ONNXRUNTIME_ROOT}/lib)

# Include directory for your headers (DA2Network.hpp)
include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable( vidDisplay 
    src/vidDisplay.cpp 
    src/filter.cpp
    src/faceDetect.cpp
)
target_link_libraries( vidDisplay ${OpenCV_LIBS} onnxruntime )

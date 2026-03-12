cmake_minimum_required(VERSION 3.16)
project( CS5330_Project1 )

set(CMAKE_CXX_STANDARD 17)

# OpenCV (DNN module included automatically)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# Include directory for your headers (features.h, vision.h, DA2Network.hpp, etc.)
include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable( vidDisplay
    src/vidDisplay.cpp
    src/filter.cpp
    src/utilities.cpp
)
target_link_libraries( vidDisplay ${OpenCV_LIBS} )

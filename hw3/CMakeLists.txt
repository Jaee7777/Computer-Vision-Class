cmake_minimum_required(VERSION 3.16)
project( CS5330_Project1 )

set(CMAKE_CXX_STANDARD 17)

# OpenCV
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )


# Include directory for your headers (DA2Network.hpp)
include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable( hw3 src/makeHist.cpp )
target_link_libraries( hw3 ${OpenCV_LIBS} )
cmake_minimum_required(VERSION 3.16)
project( CS5330_Project1 )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
link_libraries( ${OpenCV_LIBS} )

add_executable( imgDisplay src/imgDisplay.cpp )

add_executable( vidDisplay 
    src/vidDisplay.cpp 
    src/filter.cpp
)

add_executable( timeBlur
    src/timeBlur.cpp 
    src/filter.cpp
)
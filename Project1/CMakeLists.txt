cmake_minimum_required(VERSION 3.16)
project( CS5330_Project1 )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
link_libraries( ${OpenCV_LIBS} )

set(PROGRAMS
    imgDisplay
    vidDisplay
    )

foreach(PROG ${PROGRAMS})
    add_executable( ${PROG} src/${PROG}.cpp)
endforeach()

cmake_minimum_required(VERSION 3.0.2)
set(CMAKE_OSX_SYSROOT /Library/Developer/CommandLineTools/SDKs/MacOSX10.15.sdk)
set(CMAKE_MACOSX_RPATH 1)

project(opencvTest)

find_package(OpenCV REQUIRED COMPONENTS opencv_core opencv_highgui opencv_imgproc opencv_videoio)
message(STATUS "OpenCV library status:")
message(STATUS "version: ${OpenCV_VERSION}")
message(STATUS "libraries: ${OpenCV_LIBS}")
message(STATUS "include path: ${OpenCV_INCLUDE_DIRS}")
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(dlib REQUIRED)
include_directories(/usr/X11R6/lib/libX11.dylib)
link_directories(
	/opt/X11/lib/
	)


set(LINK_LIBS
	${OpenCV_LIBS}
	dlib::dlib
	X11 pthread
	)

file(COPY ${PROJECT_SOURCE_DIR}/usedfile DESTINATION ./)
add_executable(p1 p1.cpp)
target_link_libraries(p1 ${LINK_LIBS})


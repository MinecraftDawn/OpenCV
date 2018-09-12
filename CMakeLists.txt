cmake_minimum_required(VERSION 3.10)
project(OpenCV)

set(CMAKE_CXX_STANDARD 11)

add_executable(OpenCV main.cpp practice/ShowImage.h)

include_directories(C:\\opencv\\build\\include)

target_link_libraries(OpenCV C:\\opencv\\build\\x64\\vc15\\lib\\opencv_world343d.lib)


cmake_minimum_required(VERSION 2.8)
project(FFT_demo)
add_executable(FFT_demo fft.c)
target_link_libraries(FFT_demo -lm)
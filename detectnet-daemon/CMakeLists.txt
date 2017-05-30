
file(GLOB detectnetDaemonSources *.cpp)
file(GLOB detectnetDaemonIncludes *.h )

find_package( OpenCV REQUIRED )

cuda_add_executable(detectnet-daemon ${detectnetDaemonSources})
target_link_libraries(detectnet-daemon nvcaffe_parser nvinfer jetson-inference ${OpenCV_LIBS})

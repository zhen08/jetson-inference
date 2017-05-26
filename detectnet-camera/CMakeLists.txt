
file(GLOB detectnetCameraSources *.cpp)
file(GLOB detectnetCameraIncludes *.h )

find_package( OpenCV REQUIRED )

cuda_add_executable(detectnet-camera ${detectnetCameraSources})
target_link_libraries(detectnet-camera nvcaffe_parser nvinfer jetson-inference ${OpenCV_LIBS} )


CMAKE_MINIMUM_REQUIRED(VERSION 3.5)
PROJECT(detect_test)
SET(CMAKE_CXX_STANDARD 11)
INCLUDE_DIRECTORIES(
    #OpenVINO推理引擎的头文件
    /opt/intel/openvino/deployment_tools/inference_engine/include/
    /opt/intel/openvino/deployment_tools/ngraph/include/
)

FIND_PACKAGE(OpenCV REQUIRED)
SET(InferenceEngine_DIR "/opt/intel/openvino/deployment_tools/inference_engine/share")
FIND_PACKAGE(InferenceEngine)
SET(ngraph_DIR "/opt/intel/openvino/deployment_tools/ngraph")
FIND_PACKAGE(ngraph REQUIRED)
SET(ngraph_LIBRARIES "/opt/intel/openvino/deployment_tools/ngraph/lib/libngraph.so")
SET(ngraph_INCLUDE_DIRS "/opt/intel/openvino/deployment_tools/ngraph/include/")


# 编译detector.so
ADD_LIBRARY (detector SHARED ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
TARGET_INCLUDE_DIRECTORIES(detector
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
        PUBLIC ${OpenCV_INCLUDE_DIR}
        PUBLIC ${InferenceEngine_INCLUDE_DIRS}
)

TARGET_LINK_LIBRARIES(detector
        ${OpenCV_LIBS}
        ${InferenceEngine_LIBRARIES}
        ${ngraph_LIBRARIES}
)

ADD_EXECUTABLE(main main.cpp)
TARGET_LINK_LIBRARIES(main
        ${OpenCV_LIBS}
        detector
)
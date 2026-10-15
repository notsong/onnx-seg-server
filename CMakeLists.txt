cmake_minimum_required(VERSION 3.20)
project(onnx_seg_server VERSION 0.1.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(ONNXRUNTIME_ROOT "${CMAKE_SOURCE_DIR}/deps/onnxruntime-win-x64-1.22.0"
    CACHE PATH "ONNX Runtime CPU package directory")
if(WIN32 AND NOT OpenCV_DIR AND EXISTS "${CMAKE_SOURCE_DIR}/deps/opencv/build/x64/vc16/lib/OpenCVConfig.cmake")
    set(OpenCV_DIR "${CMAKE_SOURCE_DIR}/deps/opencv/build/x64/vc16/lib" CACHE PATH "OpenCV package directory")
endif()
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs dnn)
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    HINTS "${ONNXRUNTIME_ROOT}/include" REQUIRED)
find_library(ONNXRUNTIME_LIBRARY onnxruntime
    HINTS "${ONNXRUNTIME_ROOT}/lib" REQUIRED)

add_executable(onnx-seg-server src/main.cpp src/segmenter.cpp)
target_include_directories(onnx-seg-server PRIVATE
    src third_party "${ONNXRUNTIME_INCLUDE_DIR}" "${OpenCV_INCLUDE_DIRS}")
target_link_libraries(onnx-seg-server PRIVATE "${ONNXRUNTIME_LIBRARY}" "${OpenCV_LIBS}")
if(MSVC)
    target_compile_options(onnx-seg-server PRIVATE /W4 /utf-8 /permissive-)
    target_compile_definitions(onnx-seg-server PRIVATE
        NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(onnx-seg-server PRIVATE -Wall -Wextra -Wpedantic)
    find_package(Threads REQUIRED)
    target_link_libraries(onnx-seg-server PRIVATE Threads::Threads)
endif()

if(WIN32)
    file(GLOB ORT_DLLS "${ONNXRUNTIME_ROOT}/lib/*.dll")
    foreach(dll IN LISTS ORT_DLLS)
        add_custom_command(TARGET onnx-seg-server POST_BUILD COMMAND
            "${CMAKE_COMMAND}" -E copy_if_different "${dll}" "$<TARGET_FILE_DIR:onnx-seg-server>")
    endforeach()
    if(EXISTS "${CMAKE_SOURCE_DIR}/deps/opencv/build/x64/vc16/bin/opencv_world4120.dll")
        add_custom_command(TARGET onnx-seg-server POST_BUILD COMMAND
            "${CMAKE_COMMAND}" -E copy_if_different
            "${CMAKE_SOURCE_DIR}/deps/opencv/build/x64/vc16/bin/opencv_world4120.dll"
            "$<TARGET_FILE_DIR:onnx-seg-server>")
    endif()
endif()

# Deploy the executable and runtime DLLs next to config/ and models/.
install(TARGETS onnx-seg-server RUNTIME DESTINATION .)
if(WIN32)
    install(FILES ${ORT_DLLS} DESTINATION .)
    if(EXISTS "${CMAKE_SOURCE_DIR}/deps/opencv/build/x64/vc16/bin/opencv_world4120.dll")
        install(FILES "${CMAKE_SOURCE_DIR}/deps/opencv/build/x64/vc16/bin/opencv_world4120.dll"
                DESTINATION .)
    endif()
    set(CMAKE_INSTALL_SYSTEM_RUNTIME_DESTINATION .)
    include(InstallRequiredSystemLibraries)
endif()

cmake_minimum_required(VERSION 3.20)
project(BM3DMetalDylib LANGUAGES OBJCXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(bm3dmetal SHARED
    bm3dmetal.mm
)

target_include_directories(bm3dmetal PRIVATE ${VAPOURSYNTH_INCLUDE_DIRECTORY})

set_target_properties(bm3dmetal PROPERTIES 
    CXX_FLAGS "-Wno-c23-extensions -mmacosx-version-min=12.0"
    COMPILE_FLAGS "-fobjc-arc"
)

find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)

target_link_libraries(bm3dmetal PUBLIC
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
    ${ACCELERATE_FRAMEWORK}
)

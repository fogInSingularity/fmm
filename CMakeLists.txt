cmake_minimum_required(VERSION 3.21)

project(fmm C)

option(LIB_USE_NAIVE "Use naive impl in the library" OFF)
option(LIB_USE_TRANSPOSE "Use transpose impl in the library" OFF)

add_library(${PROJECT_NAME})

if(NAIVE)
    target_compile_definitions(${PROJECT_NAME} PRIVATE NAIVE)
    message(STATUS "naive impl chosen")
elseif(TRANSPOSE)
    target_compile_definitions(${PROJECT_NAME} PRIVATE TRANSPOSE)
    message(STATUS "transpose impl chosen")
elseif(KERNEL)
    target_compile_definitions(${PROJECT_NAME} PRIVATE KERNEL)
    message(STATUS "kernel impl chosen")
else()
    message(STATUS "no option provided, defaults to naive")
endif()

include(defaults.cmake)

target_sources(${PROJECT_NAME}
    PRIVATE
        src/fmm.c
)

target_include_directories(${PROJECT_NAME}
    PUBLIC
        include
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        fmm-defaults
)

add_subdirectory(benchmarks)
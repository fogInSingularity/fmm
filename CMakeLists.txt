cmake_minimum_required(VERSION 3.21)

project(fmm C)

include(defaults.cmake)

add_library(${PROJECT_NAME})

target_sources(${PROJECT_NAME}
    PRIVATE
        fmm.c
)

target_include_directories(${PROJECT_NAME}
    PUBLIC
        .
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        fmm-defaults
)

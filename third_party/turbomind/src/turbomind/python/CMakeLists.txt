# Copyright (c) OpenMMLab. All rights reserved.

cmake_minimum_required(VERSION 3.11)
project(_turbomind LANGUAGES CXX CUDA)

find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    execute_process(
      COMMAND "pybind11-config" "--cmakedir"
      RESULT_VARIABLE _COMMAND_SUCCESS
      OUTPUT_VARIABLE pybind11_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
    )
    if(_COMMAND_SUCCESS EQUAL 0 AND NOT pybind11_DIR STREQUAL "")
      list(APPEND CMAKE_PREFIX_PATH "${pybind11_DIR}")
      find_package(pybind11 CONFIG QUIET)
    endif()
endif()

if(NOT pybind11_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.13.6
    GIT_SHALLOW ON
    GIT_PROGRESS TRUE
    USES_TERMINAL_DOWNLOAD TRUE
  )
  FetchContent_MakeAvailable(pybind11)
endif()

if(NOT COMMAND pybind11_add_module)
  message(FATAL_ERROR "pybind11_add_module is unavailable after pybind11 setup.")
endif()

pybind11_add_module(${PROJECT_NAME} bind.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE turbomind xgrammar)

pybind11_add_module(_xgrammar xgrammar_bind.cpp)
target_link_libraries(_xgrammar PRIVATE core xgrammar)
target_compile_features(_xgrammar PRIVATE cxx_std_14)

if (CALL_FROM_SETUP_PY)
  string(REPLACE "." ";" _ver ${CMAKE_CUDA_COMPILER_VERSION})
  list(GET _ver 0 CUDA_MAJOR)

  if(CUDA_MAJOR GREATER_EQUAL "13")
    set(_INSTALL_CUDA_RPATH
        "\$ORIGIN"
        "\$ORIGIN/../../nvidia/nccl/lib/"
        "\$ORIGIN/../../nvidia/cu${CUDA_MAJOR}/lib/"
    )
  else()
    set(_INSTALL_CUDA_RPATH
        "\$ORIGIN"
        "\$ORIGIN/../../nvidia/nccl/lib/"
        "\$ORIGIN/../../nvidia/cuda_runtime/lib/"
        "\$ORIGIN/../../nvidia/cublas/lib/"
        "\$ORIGIN/../../nvidia/curand/lib/"
    )
  endif()
  set_target_properties(${PROJECT_NAME} PROPERTIES
      BUILD_RPATH "\$ORIGIN"
      INSTALL_RPATH "${_INSTALL_CUDA_RPATH}"
  )
endif ()

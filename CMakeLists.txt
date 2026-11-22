cmake_minimum_required(VERSION 3.20)
project(cg_poisson LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(WARNINGS $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic> $<$<CXX_COMPILER_ID:MSVC>:/W4>)

add_library(poisson src/poisson.cpp)
target_include_directories(poisson PUBLIC include)
target_compile_options(poisson PRIVATE ${WARNINGS})

add_executable(poisson_tests tests/test_poisson.cpp)
target_link_libraries(poisson_tests PRIVATE poisson)

add_executable(convergence apps/convergence.cpp)
target_link_libraries(convergence PRIVATE poisson)

enable_testing()
add_test(NAME poisson_tests COMMAND poisson_tests)

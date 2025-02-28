# Minimum CMake version and project setup
cmake_minimum_required(VERSION 3.10)
project(nanobind_solver)

# Ensure C++17 is enabled
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Explicitly add the C++17 flag for MinGW
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")


# Find Python development libraries
find_package(Python REQUIRED COMPONENTS Development)

# Set Eigen3 include directory manually (adjust the path if necessary)
set(Eigen3_INCLUDE_DIR "C:/Eigen-3/eigen-3.4.0")
include_directories(${Eigen3_INCLUDE_DIR})

# Set nanobind include directory manually (adjust the path if necessary)
set(nanobind_INCLUDE_DIR "C:/nanobind/include")
include_directories(${nanobind_INCLUDE_DIR})

# Add the solver library (creating a Python module)
add_library(solver MODULE nanobind_solver.cpp)

# Link Python and Eigen3 to the solver module
target_include_directories(solver PRIVATE ${Python_INCLUDE_DIRS} ${Eigen3_INCLUDE_DIR})
target_link_libraries(solver PRIVATE ${Python_LIBRARIES})

# Set module output properties (for Python)
set_target_properties(solver PROPERTIES PREFIX "" SUFFIX ".so")  # Adjust for your OS

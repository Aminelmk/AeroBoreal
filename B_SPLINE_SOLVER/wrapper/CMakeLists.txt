cmake_minimum_required(VERSION 3.18)
project(BSpline_solver)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if (MSVC)
    # Optimisation maximale pour la release
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /Ob2 /DNDEBUG")
    # Optionnel : désactiver certaines warnings inutiles
    add_definitions(/D_CRT_SECURE_NO_WARNINGS)
else()
    # Pour GCC/Clang
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -DNDEBUG")
endif()
# ------------------------------------------------------------
# Python and nanobind
# ------------------------------------------------------------
set(Python3_ROOT_DIR "C:/Users/nadir/AppData/Local/Programs/Python/Python313")
find_package(Python3 REQUIRED COMPONENTS Development)

include_directories(${Python3_INCLUDE_DIRS})
link_directories(${Python3_LIBRARY_DIRS})

# tsl::robin_map (nécessaire pour nanobind)
include_directories("C:/Users/nadir/Desktop/CPP/tsl-robin-map/include")

# Nanobind include
set(NANOBIND_DIR "${CMAKE_SOURCE_DIR}/nanobind")
include_directories(${NANOBIND_DIR}/include)

add_library(nanobind STATIC
    ${NANOBIND_DIR}/src/common.cpp
    ${NANOBIND_DIR}/src/error.cpp
    ${NANOBIND_DIR}/src/nb_type.cpp
    ${NANOBIND_DIR}/src/nb_func.cpp
    ${NANOBIND_DIR}/src/nb_internals.cpp
    ${NANOBIND_DIR}/src/nb_ndarray.cpp
    ${NANOBIND_DIR}/src/nb_static_property.cpp
    ${NANOBIND_DIR}/src/implicit.cpp
)

# ------------------------------------------------------------
# Include paths for your libs (Eigen, Ceres, glog)
# ------------------------------------------------------------
include_directories(
    ${Python3_INCLUDE_DIRS}
    "C:/Users/nadir/Desktop/CPP/eigen-3.4.0"
    "C:/Users/nadir/Desktop/CPP/B-Spline/ceres-install/include"
    "C:/Users/nadir/Desktop/CPP/B-Spline/glog-install/include"
)

link_directories(
    "C:/Users/nadir/Desktop/CPP/B-Spline/ceres-install/lib"
    "C:/Users/nadir/Desktop/CPP/B-Spline/glog-install/lib"
)

set(SOURCES
    bspline_wrapper.cpp
    
)

# ------------------------------------------------------------
# Build the Python module
# ------------------------------------------------------------
add_library(BSpline_solver MODULE ${SOURCES})

target_compile_definitions(BSpline_solver PRIVATE NANOBIND_STATIC_DEFINE)

target_link_libraries(BSpline_solver PRIVATE
    nanobind
    ${Python3_LIBRARIES}
    ceres
    glog
)

file(GLOB ABSEIL_LIBS "C:/Users/nadir/Desktop/CPP/B-Spline/ceres-install/lib/absl_*.lib")
target_link_libraries(BSpline_solver PRIVATE ${ABSEIL_LIBS})

# Propriétés du module Python
set_target_properties(BSpline_solver PROPERTIES
    PREFIX ""
    SUFFIX ".pyd"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

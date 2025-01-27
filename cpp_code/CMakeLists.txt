cmake_minimum_required(VERSION 3.12)
project(ProcessData)

# Spécifiez le chemin de pybind11 installé via pip
set(pybind11_DIR "/home/aminelmk/Bureau/HTML/flask_env/lib/python3.12/site-packages/pybind11/share/cmake/pybind11")

# Trouver pybind11
find_package(pybind11 REQUIRED)

# Ajouter le fichier C++ et le lier avec pybind11
add_library(process_data MODULE process_data.cpp)
target_link_libraries(process_data PRIVATE pybind11::module)

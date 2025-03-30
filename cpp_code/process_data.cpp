#include <iostream>

double process_data(double value1, double value2, double value3) {
    return value1 + value2 + value3; 
}

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(process_data, m) {
    m.def("process_data", &process_data, "Process 3 values and return their sum");
}
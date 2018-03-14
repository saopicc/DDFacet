#pragma once

#include "pybind11/include/pybind11/pybind11.h"
#include "pybind11/include/pybind11/numpy.h"
#include "gridder.h"
#include "degridder.h"
#include <cstdint>

namespace {

template <typename T> bool contains(const std::vector<T>& Vec, const T &Element)
  { return find(Vec.begin(), Vec.end(), Element) != Vec.end(); }

}

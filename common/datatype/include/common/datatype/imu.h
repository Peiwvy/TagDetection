#pragma once

#include "reflcpp/core.hpp"
#include "reflcpp/serialization.hpp"
#include "reflcpp/yaml.hpp"
#include "reflcpp/runtime.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iomanip>
#include <iostream>
#include <numeric>

namespace datatype {

struct IMU {
  double             time;
  Eigen::Vector3d    acc;
  Eigen::Vector3d    gyr;
  Eigen::Quaterniond rot;
};

}  // namespace datatype

REFLCPP_METAINFO(datatype::IMU, , (time)(acc)(gyr)(rot))

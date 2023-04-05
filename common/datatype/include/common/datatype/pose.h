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

struct Pose {
  Eigen::Vector3d    trans;
  Eigen::Quaterniond rot;
};

}  // namespace datatype

REFLCPP_METAINFO(datatype::Pose, , (trans)(rot))

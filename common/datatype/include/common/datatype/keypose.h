#pragma once

#include "reflcpp/core.hpp"
#include "reflcpp/serialization.hpp"
#include "reflcpp/yaml.hpp"
#include "reflcpp/runtime.hpp"

#include "pose.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iomanip>
#include <iostream>
#include <numeric>

namespace datatype {

struct KeyPose : Pose {
  double       time  = 0.0;
  unsigned int index = 0;
};

}  // namespace datatype

REFLCPP_METAINFO(datatype::KeyPose, (datatype::Pose), (time)(index))

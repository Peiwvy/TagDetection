#pragma once
// mylib
#include "apriltag_manager.h"
#include "common/datatype/pose.h"

#include "reflcpp/core.hpp"
#include "reflcpp/yaml.hpp"

#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/range_image/range_image.h>
#include <pcl/visualization/common/float_image_utils.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <yaml-cpp/yaml.h>

#include <memory>
#include <mutex>
#include <queue>
#include <string>

using PointType = pcl::PointXYZ;

// c++ queue for holding pointclouds based on integration size
template <typename T, typename Container = std::deque<T>> class IterableQueue : public std::queue<T, Container> {
 public:
  using iterator       = typename Container::iterator;
  using const_iterator = typename Container::const_iterator;

  iterator begin() {
    return this->c.begin();
  }
  iterator end() {
    return this->c.end();
  }
  [[nodiscard]] const_iterator begin() const {
    return this->c.begin();
  }
  [[nodiscard]] const_iterator end() const {
    return this->c.end();
  }
};

struct Outcome {
  int     id;
  cv::Mat gray;

  cv::Mat R;
  cv::Mat T;

  pcl::PointCloud<pcl::PointXYZ>::Ptr pts_ob;
  pcl::PointCloud<pcl::PointXYZ>::Ptr pts_tag;

  bool update = false;
};

struct TagDetectionParam {
  int    integration_size         = 20;
  float  angular_resolution_x_deg = 0.05;
  float  angular_resolution_y_deg = 0.05;
  float  max_angular_width_deg    = 70;
  float  max_angular_height_deg   = 70;
  double image_threshold          = 80;
  bool   add_blur                 = false;
};
REFLCPP_METAINFO(
  TagDetectionParam,
  ,
  (integration_size)(angular_resolution_x_deg)(angular_resolution_y_deg)(max_angular_width_deg)(max_angular_height_deg)(image_threshold)(add_blur))
REFLCPP_YAML(TagDetectionParam)

class TagDetection {

 public:
  /****************************************/
  /*      construct from config file      */
  /****************************************/
  explicit TagDetection(const std::string& config_file);
  /****************************************/
  /*      sensor data feed functions      */
  /****************************************/
  void feed_pointcloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr pcl);
  /****************************************/
  /*        main process function         */
  /****************************************/
  void process();
  void detect_tag(const std::vector<std::vector<float>>& points);
  void pose_estimation_3d3d(const std::vector<cv::Point3f>&, const std::vector<cv::Point3f>&, cv::Mat&, cv::Mat&);

 private:
  std::mutex pointcloud_ptr_mtx_;

  /****************************************/
  /*      submodule  class                */
  /****************************************/
  TagDetectionParam                tag_detection_param_;
  std::unique_ptr<ApriltagManager> ap_ptr_;

  IterableQueue<pcl::PointCloud<pcl::PointXYZI>::Ptr> pcq_;

  std::unordered_map<int, std::vector<cv::Point3f>> tag_map_;

  void vector_to_pcl(const std::vector<cv::Point3f>& pts, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);

 public:
  Outcome this_outcome;

  datatype::Pose thispose;
};
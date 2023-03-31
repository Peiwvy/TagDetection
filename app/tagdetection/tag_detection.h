#pragma once

#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <string>

// opencv imports
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

// pcl imports
#include <pcl/console/parse.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/png_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/range_image/range_image.h>
#include <pcl/visualization/common/float_image_utils.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "apriltag_manager.h"
#include "yaml-cpp/yaml.h"

typedef pcl::PointXYZ PointType;

// c++ queue for holding pointclouds based on integration size
template <typename T, typename Container = std::deque<T>> class iterable_queue : public std::queue<T, Container> {
 public:
  typedef typename Container::iterator       iterator;
  typedef typename Container::const_iterator const_iterator;

  iterator begin() {
    return this->c.begin();
  }
  iterator end() {
    return this->c.end();
  }
  const_iterator begin() const {
    return this->c.begin();
  }
  const_iterator end() const {
    return this->c.end();
  }
};

struct Outcome {
  int     id;
  cv::Mat gray;

  cv::Mat R;
  cv::Mat T;

  std::vector<cv::Point3f> pts_ob;
  std::vector<cv::Point3f> pts_tag;
};

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
  void               process();
  void               detect_tag(const std::vector<std::vector<float>>& points);
  void               pose_estimation_3d3d(const std::vector<cv::Point3f>&, const std::vector<cv::Point3f>&, cv::Mat&, cv::Mat&);
  Eigen::Quaterniond rotation_to_quaternion(Eigen::Matrix3d);

 private:
  std::mutex pointcloud_ptr_mtx_;

  std::queue<std::string> rst_queue_;
  std::mutex              rst_mtx_;

  /****************************************/
  /*      submodule  class                */
  /****************************************/

  iterable_queue<pcl::PointCloud<pcl::PointXYZI>::Ptr> pcq;

  std::unique_ptr<ApriltagManager> ap_ptr_;

  std::unordered_map<int, std::vector<cv::Point3f>> tag_map;

  // alogrithm parameters
  std::string lidar_topic;
  int         integration_size = 20;
  float       angular_resolution_x_deg;
  float       angular_resolution_y_deg;
  float       max_angular_width_deg;
  float       max_angular_height_deg;
  double      image_threshold;
  bool        add_blur;

 public:
  cv::Mat R_glo;
  cv::Mat T_glo;

  bool    count = false;
  cv::Mat gray_glo;

  std::vector<cv::Point3f> pts_ob_glo;
  std::vector<cv::Point3f> pts_tag_glo;

  std::optional<Outcome> this_outcome;
  // outpu
};
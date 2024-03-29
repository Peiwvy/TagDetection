// app
#include "tag_detection.h"

#include "tools/rosbag.h"

#include "reflcpp/core.hpp"
#include "reflcpp/yaml.hpp"

#include "timer.h"
// ros
#include <cv_bridge/cv_bridge.h>
#include <pcl_ros/point_cloud.h>

#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>

#include <ros/callback_queue.h>
#include <ros/ros.h>
// standard c++
#include <filesystem>
#include <memory>
#include <utility>

namespace fs = std::filesystem;

/****************************************/
/*            user app here             */
/****************************************/
std::unique_ptr<TagDetection> app;

/****************************************/
/*            callback here             */
/*    constptr for better performance   */
/****************************************/
void lidar_cb(const sensor_msgs::PointCloud2& cloud_msg) {
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>());
  pcl::fromROSMsg(cloud_msg, *cloud);
  app->feed_pointcloud(cloud);
}

void publish_pose(const ros::Publisher& pub, const Eigen::Matrix3d& r, const Eigen::Vector3d& t) {
  geometry_msgs::PoseStamped pose_stamped;
  pose_stamped.header.frame_id = "livox_frame";
  pose_stamped.header.stamp    = ros::Time::now();

  pose_stamped.pose.position.x = t(0);
  pose_stamped.pose.position.y = t(1);
  pose_stamped.pose.position.z = t(2);

  auto pose_rotation_new = Eigen::Quaterniond(r);
  pose_rotation_new.normalize();

  pose_stamped.pose.orientation.x = pose_rotation_new.x();
  pose_stamped.pose.orientation.y = pose_rotation_new.y();
  pose_stamped.pose.orientation.z = pose_rotation_new.z();
  pose_stamped.pose.orientation.w = pose_rotation_new.w();

  pub.publish(pose_stamped);
}

void publish_tag_line(const ros::Publisher& pub, const Eigen::Vector3d& t) {
  visualization_msgs::Marker marker;
  marker.header.frame_id = "livox_frame";                           // 坐标系
  marker.header.stamp    = ros::Time::now();                        // 时间戳
  marker.ns              = "line";                                  // 命名空间
  marker.id              = 0;                                       // ID
  marker.type            = visualization_msgs::Marker::LINE_STRIP;  // 类型为直线
  marker.action          = visualization_msgs::Marker::ADD;

  marker.color.r = 1.0;
  marker.color.g = 0.0;
  marker.color.b = 0.0;
  marker.color.a = 1.0;

  marker.scale.x = 0.1;  // 箭头的长度
  marker.scale.y = 0.1;  // 箭头的宽度
  marker.scale.z = 0.1;  // 箭头的宽度

  marker.pose.position.x    = 0;
  marker.pose.position.y    = 0;
  marker.pose.position.z    = 0;
  marker.pose.orientation.x = 0.0;
  marker.pose.orientation.y = 0.0;
  marker.pose.orientation.z = 0.0;
  marker.pose.orientation.w = 1.0;

  geometry_msgs::Point p;
  p.x = 0;
  p.y = 0;
  p.z = 0;
  marker.points.push_back(p);
  p.x = t(0);
  p.y = t(1);
  p.z = t(2);
  marker.points.push_back(p);

  pub.publish(marker);
}

void publish_tag_cloud(const ros::Publisher& pub, const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
  sensor_msgs::PointCloud2 pc_out;
  pcl::toROSMsg(*cloud, pc_out);

  pc_out.header.frame_id = "livox_frame";
  pc_out.header.stamp    = ros::Time::now();

  pub.publish(pc_out);
}

void publish_image(const ros::Publisher& pub, const cv::Mat& image) {
  sensor_msgs::ImagePtr image_msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", image).toImageMsg();
  image_msg->header.frame_id      = "livox_frame";
  image_msg->header.stamp         = ros::Time::now();
  pub.publish(image_msg);
}

/****************************************/
/*        program configs struct        */
/****************************************/
struct ProgramConfigs {
  // rosbag
  std::string bag_file;
  float       bag_rate;
  // app configuration file path
  std::string app_config;
  // rostopic
  std::string lidar_topic;
  // log
  std::string log_file;
};
REFLCPP_METAINFO(ProgramConfigs, , (bag_file)(bag_rate)(app_config)(lidar_topic)(log_file))
REFLCPP_YAML(ProgramConfigs)

void as_absolute_path(std::string& path, const std::string& prefix) {
  if (!path.empty() && fs::path(path).is_relative()) {
    auto path_temp = fs::path(prefix) / path;
    path           = path_temp.lexically_normal();
  }
}

int main(int argc, char* argv[]) {
  /****************************************/
  /*      configure from command line     */
  /****************************************/
  std::string config_file;
  if (argc == 1) {
    config_file = fs::path(WORK_SPACE_PATH) / "config/ros_configs.yaml";
  } else if (argc == 2) {
    config_file = argv[1];
  } else {
    ROS_FATAL_STREAM("usage: " << argv[0] << " CONFIG_FILE.");
    return -1;
  }
  ROS_INFO_STREAM("configuring from " << config_file);

  auto prog_cfg    = YAML::LoadFile(config_file).as<ProgramConfigs>();
  auto prefix_path = fs::path(config_file).parent_path();
  as_absolute_path(prog_cfg.bag_file, prefix_path);
  as_absolute_path(prog_cfg.app_config, prefix_path);
  as_absolute_path(prog_cfg.log_file, prefix_path);
  /****************************************/
  /*            configure app             */
  /****************************************/
  app = std::make_unique<TagDetection>(prog_cfg.app_config);

  /****************************************/
  /*            configure ros             */
  /****************************************/
  ros::init(argc, argv, "ros_example");
  ros::NodeHandle nh;

  auto lidar_sub = nh.subscribe(prog_cfg.lidar_topic, 5, lidar_cb);

  auto iilfm_pose_pub        = nh.advertise<geometry_msgs::PoseStamped>("/iilfm/pose", 10);
  auto iilfm_feature_pub     = nh.advertise<sensor_msgs::PointCloud2>("/iilfm/features", 10);
  auto iilfm_gray_pub        = nh.advertise<sensor_msgs::Image>("/iilfm/gray_image", 10);
  auto iilfm_otigin2pose_pub = nh.advertise<visualization_msgs::Marker>("/visualization_marker", 10);

  /****************************************/
  /*            configure bag             */
  /****************************************/
  BagPlayer bag_player(nh);
  if (!prog_cfg.bag_file.empty()) {
    bag_player.open(prog_cfg.bag_file);
    bag_player.set_queue_size(1);
    bag_player.set_rate(prog_cfg.bag_rate);  // set to 0 for full speed (not recommend)
  }

  /****************************************/
  /*              main loop               */
  /****************************************/
  ROS_INFO_STREAM("begin loop.");
  auto callbacks = ros::getGlobalCallbackQueue();
  while (ros::ok()) {
    // play rosbag
    if (bag_player.is_open()) {
      // end loop if bag end
      if (bag_player.eof())
        ros::shutdown();
      bag_player.play_once();
    }
    // process ros message callback (sleep forever if no callbacks available)
    if (bag_player.is_open()) {
      callbacks->callAvailable();
    } else {
      callbacks->callAvailable(ros::WallDuration(999));
    }

    if (app->this_outcome.update) {
      publish_tag_line(iilfm_otigin2pose_pub, app->this_outcome.T);
      publish_tag_cloud(iilfm_feature_pub, app->this_outcome.pts_tag);
      publish_image(iilfm_gray_pub, app->this_outcome.gray);
      publish_pose(iilfm_pose_pub, app->this_outcome.R, app->this_outcome.T);
      app->this_outcome.update = false;
    }
  }

  /****************************************/
  /*    (optional): some post process     */
  /****************************************/
  Timer::PrintAll();
  Timer::DumpIntoFile(prog_cfg.log_file);

  return 0;
}

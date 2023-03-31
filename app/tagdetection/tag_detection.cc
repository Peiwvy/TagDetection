#include "tag_detection.h"

#include <math.h>

TagDetection::TagDetection(const std::string& config_file) {
  auto conf = YAML::LoadFile(config_file);

  integration_size         = conf["integration_size"].as<int>();
  angular_resolution_x_deg = conf["angular_resolution_x_deg"].as<float>();
  angular_resolution_y_deg = conf["angular_resolution_y_deg"].as<float>();
  max_angular_width_deg    = conf["max_angular_width_deg"].as<float>();
  max_angular_height_deg   = conf["max_angular_height_deg"].as<float>();
  image_threshold          = conf["image_threshold"].as<double>();
  add_blur                 = conf["add_blur"].as<bool>();

  auto tag_family       = conf["tag_family"].as<std::string>();
  auto tag_decimate     = conf["tag_decimate"].as<double>();
  auto tag_blur         = conf["tag_blur"].as<double>();
  auto tag_threads      = conf["tag_threads"].as<int>();
  auto tag_debug        = conf["tag_debug"].as<bool>();
  auto tag_refine_edges = conf["tag_refine_edges"].as<bool>();

  conf = conf["tag_list"];
  for (YAML::const_iterator it = conf.begin(); it != conf.end(); ++it) {
    YAML::Node attributes = it->second;
    YAML::Node tagid      = attributes["id"];

    YAML::Node firstv  = attributes["firstvertex"];
    YAML::Node secondv = attributes["secondvertex"];
    YAML::Node thirdv  = attributes["thirdvertex"];
    YAML::Node fourthv = attributes["fourthvertex"];

    std::vector<cv::Point3f> pts0;
    pts0.emplace_back(firstv[0].as<float>(), firstv[1].as<float>(), firstv[2].as<float>());
    pts0.emplace_back(secondv[0].as<float>(), secondv[1].as<float>(), secondv[2].as<float>());
    pts0.emplace_back(thirdv[0].as<float>(), thirdv[1].as<float>(), thirdv[2].as<float>());
    pts0.emplace_back(fourthv[0].as<float>(), fourthv[1].as<float>(), fourthv[2].as<float>());

    tag_map.insert(std::make_pair(tagid.as<int>(), pts0));
  }

  ap_ptr_ = std::make_unique<ApriltagManager>();
  ap_ptr_->create_tag_detector(tag_family, tag_decimate, tag_blur, tag_threads, tag_debug, tag_refine_edges);
}

void TagDetection::feed_pointcloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr pcl) {
  std::unique_lock lock(pointcloud_ptr_mtx_);

  pcq.push(pcl);

  if (pcq.size() >= integration_size) {
    // vector of valid points
    std::vector<std::vector<float>> valid_points;
    std::vector<float>              valid_point;
    // iterate through the queue and create a vector of valid points
    for (const auto& cloudptr : pcq) {
      for (auto& point : cloudptr->points) {
        // ensure the point is not too close to the sensor in x
        if (point.x != 0) {
          valid_point = {point.x, point.y, point.z, point.intensity};
          valid_points.push_back(valid_point);
        }
      }
    }

    detect_tag(valid_points);
    pcq.pop();
  }
}

void TagDetection::process() {}

void TagDetection::detect_tag(const std::vector<std::vector<float>>& points) {

  // create an empty pointcloud to fill witht the valid points, and valid points
  // with intensity
  pcl::PointCloud<pcl::PointXYZ> valid_cloud;
  valid_cloud.width    = points.size();
  valid_cloud.height   = 1;
  valid_cloud.is_dense = false;
  valid_cloud.points.resize(valid_cloud.width * valid_cloud.height);

  pcl::PointCloud<pcl::PointXYZ> valid_cloud_i;
  valid_cloud_i.width    = points.size();
  valid_cloud_i.height   = 1;
  valid_cloud_i.is_dense = false;
  valid_cloud_i.points.resize(valid_cloud_i.width * valid_cloud_i.height);

  // write the valid points to the new pointcloud
  for (size_t p = 0; p < points.size(); ++p) {
    valid_cloud.points[p].x = points[p][0];
    valid_cloud.points[p].y = points[p][1];
    valid_cloud.points[p].z = points[p][2];

    valid_cloud_i.points[p].x = (points[p][3] != 0) ? points[p][0] * points[p][3] : points[p][0];
    valid_cloud_i.points[p].y = (points[p][3] != 0) ? points[p][1] * points[p][3] : points[p][1];
    valid_cloud_i.points[p].z = (points[p][3] != 0) ? points[p][2] * points[p][3] : points[p][2];
  }

  // initialize the range images
  float noise_level = 0.0;
  float min_range   = 0.0f;
  int   border_size = 1;

  pcl::RangeImage::Ptr range_image_ptr(new pcl::RangeImage);
  pcl::RangeImage&     range_image = *range_image_ptr;

  pcl::RangeImage::Ptr range_image_i_ptr(new pcl::RangeImage);
  pcl::RangeImage&     range_image_i = *range_image_i_ptr;

  auto            angular_resolution_x = (float)(angular_resolution_x_deg * (M_PI / 180.0f));
  auto            angular_resolution_y = (float)(angular_resolution_y_deg * (M_PI / 180.0f));
  auto            max_angular_width    = (float)(max_angular_width_deg * (M_PI / 180.0f));
  auto            max_angular_height   = (float)(max_angular_height_deg * (M_PI / 180.0f));
  Eigen::Affine3f sensor_pose          = (Eigen::Affine3f)Eigen::Translation3f(0.0f, 0.0f, 0.0f);

  pcl::RangeImage::CoordinateFrame coordinate_frame = pcl::RangeImage::LASER_FRAME;

  range_image.createFromPointCloud(valid_cloud,
                                   angular_resolution_x,
                                   angular_resolution_y,
                                   max_angular_width,
                                   max_angular_height,
                                   sensor_pose,
                                   coordinate_frame,
                                   noise_level,
                                   min_range,
                                   border_size);

  range_image_i.createFromPointCloud(valid_cloud_i,
                                     angular_resolution_x,
                                     angular_resolution_y,
                                     max_angular_width,
                                     max_angular_height,
                                     sensor_pose,
                                     coordinate_frame,
                                     noise_level,
                                     min_range,
                                     border_size);

  // normalize the intensity, otherwise we cannot transfer it into CV Mat
  // notice: return a pointer to a new float array containing the range values
  float* ranges = range_image_i.getRangesArray();  // this will get the intensity values

  float max = -FLT_MAX;
  for (int i = 0, n = range_image_i.width * range_image_i.height; i < n; ++i) {
    float val = *(ranges + i);
    if (val >= -FLT_MAX && val <= FLT_MAX) {
      max = std::max(max, val);
    }
  }

  // Create cv::Mat
  cv::Mat       image(range_image_i.height, range_image_i.width, CV_8UC4);
  unsigned char r, g, b;

// pcl::PointCloud to cv::Mat
#pragma omp parallel for
  for (int y = 0; y < range_image_i.height; y++) {
    for (int x = 0; x < range_image_i.width; x++) {
      pcl::PointWithRange range_pt = range_image_i.getPoint(x, y);
      // normalize
      float value = range_pt.range / max;
      // Get RGB color values for a given float in [0, 1]
      pcl::visualization::FloatImageUtils::getColorForFloat(value, r, g, b);

      image.at<cv::Vec4b>(y, x)[0] = b;
      image.at<cv::Vec4b>(y, x)[1] = g;
      image.at<cv::Vec4b>(y, x)[2] = r;
      image.at<cv::Vec4b>(y, x)[3] = 255;
    }
  }

  // create greyscale image for detections
  cv::Mat gray;
  cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  cv::threshold(gray, gray, image_threshold, 255, cv::THRESH_BINARY);

  if (add_blur) {
    cv::GaussianBlur(gray, gray, cv::Size(3, 3), 25, 0, 4);
  }

  gray_glo = gray;
  ap_ptr_->apriltag_detector_detect(gray);

  // go over the detections to get pts_ob
  std::vector<cv::Point3f> pts_ob;   // 观测二维码的顶点
  std::vector<cv::Point3f> pts_tag;  // 对应id二维码的顶点

  for (int i = 0; i < ap_ptr_->detect_num(); i++) {
    auto [this_id, these_vertex] = ap_ptr_->get(i);
    if (tag_map.find(this_id) != tag_map.end()) {
      pts_tag = tag_map[this_id];
    } else
      continue;

    for (auto& j : these_vertex) {
      auto x = j.x;
      auto y = j.y;
      // if the vertex is unobserved
      if (!range_image.isObserved(round(x), round(y))) {
        // check if there exists a pair of neighbor points that is symmetric to
        // each other with respect to the unobserved point
        for (int pixel_gap = 0; pixel_gap < 3; pixel_gap++) {
          // fix the azimuth
          if (range_image.isObserved(round(x), round(y + pixel_gap))) {
            if (range_image.isObserved(round(x), round(y - pixel_gap))) {
              pcl::PointWithRange point_up;
              pcl::PointWithRange point_down;

              range_image.calculate3DPoint(round(x), round(y + pixel_gap), point_up);
              range_image.calculate3DPoint(round(x), round(y - pixel_gap), point_down);

              float ratio      = point_down.range / point_up.range;
              float inv_ratio  = 1.0f / (1.0f + ratio);
              float down_ratio = ratio * inv_ratio;

              auto interpolate = [inv_ratio, down_ratio](float up, float down) -> float { return up * inv_ratio + down * down_ratio; };

              pts_ob.emplace_back(
                interpolate(point_up.x, point_down.x), interpolate(point_up.y, point_down.y), interpolate(point_up.z, point_down.z));
              break;
            }
          }
        }
      } else {  // if the vertex is observed
        pcl::PointWithRange range_pt_ap;
        range_image.calculate3DPoint(round(x), round(y), range_pt_ap);
        pts_ob.emplace_back(range_pt_ap.x, range_pt_ap.y, range_pt_ap.z);
      }
    }
    pts_ob_glo  = pts_ob;
    pts_tag_glo = pts_tag;
  }

  // 如何检测到，则计算相对姿态
  if (!pts_ob.empty()) {
    cv::Mat r(3, 3, CV_32FC1);
    cv::Mat t(3, 1, CV_32FC1);
    pose_estimation_3d3d(pts_ob, pts_tag, r, t);  // t,r: ob 在 tag下 坐标

    r.copyTo(R_glo);
    t.copyTo(T_glo);

    // this_outcome->R = r;
    // this_outcome->T = t;

    // this_outcome->pts_ob  = pts_ob;
    // this_outcome->pts_tag = pts_tag;

    // // this_outcome->gray = gray;
    // this_outcome->id = 0;

    count = true;
  }
}

void TagDetection::pose_estimation_3d3d(const std::vector<cv::Point3f>& pts1, const std::vector<cv::Point3f>& pts2, cv::Mat& R, cv::Mat& T) {
  cv::Point3f p1, p2;  // center of mass
  int         N = pts1.size();
  for (int i = 0; i < N; i++) {
    p1 += pts1[i];
    p2 += pts2[i];
  }
  p1 = cv::Point3f(cv::Vec3f(p1) / N);
  p2 = cv::Point3f(cv::Vec3f(p2) / N);

  std::vector<cv::Point3f> q1(N), q2(N);  // remove the center
  for (int i = 0; i < N; i++) {
    q1[i] = pts1[i] - p1;
    q2[i] = pts2[i] - p2;
  }

  // compute q1*q2^T
  Eigen::Matrix3d w = Eigen::Matrix3d::Zero();
  for (int i = 0; i < N; i++) {
    w += Eigen::Vector3d(q1[i].x, q1[i].y, q1[i].z) * Eigen::Vector3d(q2[i].x, q2[i].y, q2[i].z).transpose();
  }

  // SVD on W
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(w, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d                   u = svd.matrixU();
  Eigen::Matrix3d                   v = svd.matrixV();

  if (u.determinant() * v.determinant() < 0) {
    for (int x = 0; x < 3; ++x) {
      u(x, 2) *= -1;
    }
  }

  Eigen::Matrix3d r = u * (v.transpose());
  Eigen::Vector3d t = Eigen::Vector3d(p1.x, p1.y, p1.z) - r * Eigen::Vector3d(p2.x, p2.y, p2.z);

  // convert to cv::Mat
  R = (cv::Mat_<double>(3, 3) << r(0, 0), r(0, 1), r(0, 2), r(1, 0), r(1, 1), r(1, 2), r(2, 0), r(2, 1), r(2, 2));
  T = (cv::Mat_<double>(3, 1) << t(0, 0), t(1, 0), t(2, 0));
}

Eigen::Quaterniond TagDetection::rotation_to_quaternion(Eigen::Matrix3d R) {
  auto q = Eigen::Quaterniond(R);
  q.normalize();
  return q;
}

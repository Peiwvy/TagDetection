#include "tag_detection.h"

#include "timer.h"

#include <cmath>
#include <execution>

TagDetection::TagDetection(const std::string& config_file) {
  auto conf = YAML::LoadFile(config_file);

  tag_detection_param_ = conf.as<TagDetectionParam>();

  conf    = conf["tag_param"];
  ap_ptr_ = std::make_unique<ApriltagManager>();
  ap_ptr_->create_tag_detector(conf["tag_family"].as<std::string>(),
                               conf["tag_decimate"].as<double>(),
                               conf["tag_blur"].as<double>(),
                               conf["tag_threads"].as<int>(),
                               conf["tag_debug"].as<bool>(),
                               conf["tag_refine_edges"].as<bool>());
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

    tag_map_.insert(std::make_pair(tagid.as<int>(), pts0));
  }

  this_outcome.pts_tag.reset(new pcl::PointCloud<pcl::PointXYZ>);
  this_outcome.pts_ob.reset(new pcl::PointCloud<pcl::PointXYZ>);
}

void TagDetection::feed_pointcloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr pcl) {
  std::unique_lock lock(pointcloud_ptr_mtx_);

  pcq_.push(pcl);

  if (static_cast<int>(pcq_.size()) >= tag_detection_param_.integration_size) {
    // vector of valid points
    std::vector<std::vector<float>> valid_points;
    std::vector<float>              valid_point;

    Timer::Evaluate(
      [&, this]() {
        for (const auto& cloudptr : pcq_) {
          for (auto& point : cloudptr->points) {
            // ensure the point is not too close to the sensor in x
            if (point.x != 0) {
              valid_point = {point.x, point.y, point.z, point.intensity};
              valid_points.push_back(valid_point);
            }
          }
        }
      },
      "feed_pointcloud");
    // iterate through the queue and create a vector of valid points

    Timer::Evaluate([&, this]() { detect_tag(valid_points); }, "detect_tag");

    pcq_.pop();
  }
}

void TagDetection::process() {}

void TagDetection::detect_tag(const std::vector<std::vector<float>>& points) {
  // create empty point clouds to fill with the valid points and valid points with intensity.
  pcl::PointCloud<pcl::PointXYZ>::Ptr valid_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  valid_cloud->resize(points.size());
  pcl::PointCloud<pcl::PointXYZI>::Ptr valid_cloud_i(new pcl::PointCloud<pcl::PointXYZI>);
  valid_cloud_i->resize(points.size());

  // fill the point clouds with valid points.
  std::transform(std::execution::par, points.begin(), points.end(), valid_cloud->begin(), [](const auto& p) {
    return pcl::PointXYZ{p[0], p[1], p[2]};
  });

  std::transform(std::execution::par, points.begin(), points.end(), valid_cloud_i->begin(), [](const auto& p) {
    pcl::PointXYZI pt;
    pt.x         = (p[3] != 0) ? p[0] * p[3] : p[0];
    pt.y         = (p[3] != 0) ? p[1] * p[3] : p[1];
    pt.z         = (p[3] != 0) ? p[2] * p[3] : p[2];
    pt.intensity = p[3];
    return pt;
  });

  // initialize the range images.
  float noise_level = 0.0;
  float min_range   = 0.0f;
  int   border_size = 1;

  pcl::RangeImage::Ptr range_image_ptr(new pcl::RangeImage);
  pcl::RangeImage&     range_image = *range_image_ptr;

  pcl::RangeImage::Ptr range_image_i_ptr(new pcl::RangeImage);
  pcl::RangeImage&     range_image_i = *range_image_i_ptr;

  auto            angular_resolution_x = (float)(tag_detection_param_.angular_resolution_x_deg * (M_PI / 180.0f));
  auto            angular_resolution_y = (float)(tag_detection_param_.angular_resolution_y_deg * (M_PI / 180.0f));
  auto            max_angular_width    = (float)(tag_detection_param_.max_angular_width_deg * (M_PI / 180.0f));
  auto            max_angular_height   = (float)(tag_detection_param_.max_angular_height_deg * (M_PI / 180.0f));
  Eigen::Affine3f sensor_pose          = (Eigen::Affine3f)Eigen::Translation3f(0.0f, 0.0f, 0.0f);

  pcl::RangeImage::CoordinateFrame coordinate_frame = pcl::RangeImage::LASER_FRAME;

  range_image.createFromPointCloud(*valid_cloud,
                                   angular_resolution_x,
                                   angular_resolution_y,
                                   max_angular_width,
                                   max_angular_height,
                                   sensor_pose,
                                   coordinate_frame,
                                   noise_level,
                                   min_range,
                                   border_size);

  range_image_i.createFromPointCloud(*valid_cloud_i,
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
  float  max    = -FLT_MAX;
  for (int i = 0, n = range_image_i.width * range_image_i.height; i < n; ++i) {
    float val = *(ranges + i);
    if (val >= -FLT_MAX && val <= FLT_MAX) {
      max = std::max(max, val);
    }
  }

  // Create cv::Mat
  cv::Mat       image(range_image_i.height, range_image_i.width, CV_8UC4);
  unsigned char r = 0;
  unsigned char g = 0;
  unsigned char b = 0;

  // pcl::PointCloud to cv::Mat
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

  // convert the CV Mat to a grayscale image for detections.
  cv::Mat gray;
  cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  cv::threshold(gray, gray, tag_detection_param_.image_threshold, 255, cv::THRESH_BINARY);

  if (tag_detection_param_.add_blur) {
    cv::GaussianBlur(gray, gray, cv::Size(3, 3), 25, 0, 4);
  }

  ap_ptr_->apriltag_detector_detect(gray);

  // go over the detections to get pts_ob
  std::vector<cv::Point3f> pts_ob;   // 观测二维码的顶点
  std::vector<cv::Point3f> pts_tag;  // 对应id二维码的顶点

  for (int i = 0; i < ap_ptr_->detect_num(); i++) {
    auto [this_id, these_vertex] = ap_ptr_->get(i);
    if (tag_map_.find(this_id) != tag_map_.end()) {  // if this id is in the tag_map_
      pts_tag = tag_map_[this_id];
    } else {
      continue;
    }

    for (auto& j : these_vertex) {             // for each vertex
      auto [x, y] = std::make_pair(j.x, j.y);  // get its x and y coordinates.

      // if the vertex is unobserved.
      if (!range_image.isObserved(round(x), round(y))) {
        // check if there exists a pair of neighbor points that is symmetric to each other with
        // respect to the unobserved point.
        for (int pixel_gap = 0; pixel_gap < 3; pixel_gap++) {
          // fix the azimuth.
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
              break;  // break out of the inner loop if a pair of neighbor points was found.
            }
          }
        }
      } else {  // if the vertex is observed
        pcl::PointWithRange range_pt_ap;
        range_image.calculate3DPoint(round(x), round(y), range_pt_ap);
        pts_ob.emplace_back(range_pt_ap.x, range_pt_ap.y, range_pt_ap.z);
      }
    }
  }

  // if the position of the observed tags is computed, store the result.
  if (!pts_ob.empty()) {
    cv::Mat r(3, 3, CV_32FC1);
    cv::Mat t(3, 1, CV_32FC1);
    pose_estimation_3d3d(pts_ob, pts_tag, r, t);

    this_outcome.R = r;
    this_outcome.T = t;

    vector_to_pcl(pts_ob, this_outcome.pts_ob);
    vector_to_pcl(pts_tag, this_outcome.pts_tag);
    this_outcome.gray   = gray;
    this_outcome.id     = 0;
    this_outcome.update = true;
  }
}

void TagDetection::vector_to_pcl(const std::vector<cv::Point3f>& pts, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
  cloud->clear();
  cloud->width  = static_cast<uint32_t>(pts.size());
  cloud->height = 1;
  cloud->points.resize(cloud->width);

  std::transform(pts.begin(), pts.end(), cloud->points.begin(), [](const auto& p) { return pcl::PointXYZ{p.x, p.y, p.z}; });
}

void TagDetection::pose_estimation_3d3d(const std::vector<cv::Point3f>& pts1, const std::vector<cv::Point3f>& pts2, cv::Mat& R, cv::Mat& T) {
  const int   n  = static_cast<int>(pts1.size());
  cv::Point3f p1 = std::accumulate(pts1.begin(), pts1.end(), cv::Point3f(0.0f));
  cv::Point3f p2 = std::accumulate(pts2.begin(), pts2.end(), cv::Point3f(0.0f));
  p1             = p1 * (1.0f / n);
  p2             = p2 * (1.0f / n);

  std::vector<cv::Point3f> q1(n);
  std::vector<cv::Point3f> q2(n);
  std::transform(pts1.begin(), pts1.end(), q1.begin(), [&](const auto& p) { return p - p1; });
  std::transform(pts2.begin(), pts2.end(), q2.begin(), [&](const auto& p) { return p - p2; });

  // compute q1*q2^T
  Eigen::Matrix3d w = Eigen::Matrix3d::Zero();
  for (int i = 0; i < n; i++) {
    w += Eigen::Vector3d(q1[i].x, q1[i].y, q1[i].z) * Eigen::Vector3d(q2[i].x, q2[i].y, q2[i].z).transpose();
  }

  Eigen::JacobiSVD<Eigen::Matrix3d> svd(w, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d                   u = svd.matrixU();
  Eigen::Matrix3d                   v = svd.matrixV();

  if (u.determinant() * v.determinant() < 0) {
    u.col(2) *= -1.0;
  }

  Eigen::Matrix3d r = u * (v.transpose());
  Eigen::Vector3d t = Eigen::Vector3d(p1.x, p1.y, p1.z) - r * Eigen::Vector3d(p2.x, p2.y, p2.z);

  cv::eigen2cv(r, R);
  cv::eigen2cv(t, T);
}

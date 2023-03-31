#include "apriltag_manager.h"

ApriltagManager::ApriltagManager() {
  td_ = apriltag_detector_create();
};

apriltag_detection_t* ApriltagManager::get_old(size_t i) {
  apriltag_detection_t* det = nullptr;
  ::zarray_get(this->detections_, i, &det);
  return det;
}

auto ApriltagManager::get(size_t i) -> std::pair<int, std::vector<cv::Point2f>> {
  apriltag_detection_t* det = nullptr;
  ::zarray_get(this->detections_, i, &det);

  std::vector<cv::Point2f> points;

  for (auto& i : det->p) {
    cv::Point2f pt(i[0], i[1]);
    points.push_back(pt);
  }

  return {static_cast<size_t>(det->id), points};
}

size_t ApriltagManager::detect_num() {
  return static_cast<size_t>(zarray_size(detections_));
}

void ApriltagManager::apriltag_detector_detect(const cv::Mat& gray) {
  image_u8_t im = {.width = gray.cols, .height = gray.rows, .stride = gray.cols, .buf = gray.data};
  detections_   = ::apriltag_detector_detect(this->td_, &im);
}

void ApriltagManager::create_tag_detector(
  const std::string& tag_family, double tag_decimate, double tag_blur, int tag_threads, bool tag_debug, bool tag_refine_edges) {
  apriltag_family_t* tf = nullptr;

  if (!strcmp(tag_family.c_str(), "tag36h11")) {
    tf = tag36h11_create();
  } else if (!strcmp(tag_family.c_str(), "tag25h9")) {
    tf = tag25h9_create();
  } else if (!strcmp(tag_family.c_str(), "tag16h5")) {
    tf = tag16h5_create();
  } else if (!strcmp(tag_family.c_str(), "tagCircle21h7")) {
    tf = tagCircle21h7_create();
  } else if (!strcmp(tag_family.c_str(), "tagCircle49h12")) {
    tf = tagCircle49h12_create();
  } else if (!strcmp(tag_family.c_str(), "tagStandard41h12")) {
    tf = tagStandard41h12_create();
  } else if (!strcmp(tag_family.c_str(), "tagStandard52h13")) {
    tf = tagStandard52h13_create();
  } else if (!strcmp(tag_family.c_str(), "tagCustom48h12")) {
    tf = tagCustom48h12_create();
  } else {
    printf("Unrecognized tag family name. Use e.g. \"tag36h11\".\n");
    exit(-1);
  }

  apriltag_detector_add_family(this->td_, tf);

  this->td_->quad_decimate = tag_decimate;
  this->td_->quad_sigma    = tag_blur;
  this->td_->nthreads      = tag_threads;
  this->td_->debug         = tag_debug;
  this->td_->refine_edges  = tag_refine_edges;
}
#include <string>

#include <opencv2/core.hpp>

extern "C" {
#include "apriltag.h"
#include "common/getopt.h"
#include "tag16h5.h"
#include "tag25h9.h"
#include "tag36h11.h"
#include "tagCircle21h7.h"
#include "tagCircle49h12.h"
#include "tagCustom48h12.h"
#include "tagStandard41h12.h"
#include "tagStandard52h13.h"
}

class ApriltagManager {
 private:
  apriltag_detector_t* td_;
  zarray_t*            detections_;

 public:
  ApriltagManager();
  void
         create_tag_detector(const std::string& tag_family, double tag_decimate, double tag_blur, int tag_threads, bool tag_debug, bool tag_refine_edges);
  void   apriltag_detector_detect(const cv::Mat& gray);
  size_t detect_num();
  auto   get(size_t i) -> std::pair<int, std::vector<cv::Point2f>>;

  [[deprecated("Use the get() instead")]] apriltag_detection_t* get_old(size_t i);
};

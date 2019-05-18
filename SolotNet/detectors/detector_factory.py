from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .ctdet import CtdetDetector
from .yolodet import YoloV3Detector

detector_factory = {
  'ctdet': CtdetDetector,
  'yolodet': YoloV3Detector,
}
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .sample.ctdet import CTDetDataset
from .sample.yolo  import YOLODataset

from .coco import COCO
from .visdrone import VisDrone

dataset_factory = {
  'coco': COCO,
  'visdrone': VisDrone,
}

_sample_factory = {
  'ctdet': CTDetDataset,
  'yolo': YOLODataset,
}


def get_dataset(dataset, task):
  class Dataset(dataset_factory[dataset], _sample_factory[task]):
    pass
  return Dataset
  
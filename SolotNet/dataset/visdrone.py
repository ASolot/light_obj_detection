# import glob
# import random
# import os
# import numpy as np

# import torch

# import torch.utils.data as data
# from PIL import Image
# import torchvision.transforms as transforms

# import matplotlib.pyplot as plt
# import matplotlib.patches as patches

# from skimage.transform import resize

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os

import torch.utils.data as data

# import sys

VISDRONE_CLASSES = ('Ignored Regions', 'Pedestrian', 'People', 'Bicycle', 
                    'Car', 'Van', 'Truck', 'Tricycle', 'Awning-tricycle', 
                    'Bus', 'Motorbike', 'Other')

class VisDrone(data.Dataset):
  num_classes = 11
  default_resolution = [512, 512]
  mean = np.array([0.40789654, 0.44719302, 0.47026115],
                   dtype=np.float32).reshape(1, 1, 3)
  std  = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)

  def __init__(self, opt, split):
    super(VisDrone, self).__init__()

    self.data_dir = os.path.join(opt.data_dir, 'VISDRONE')
    self.data_dir = os.path.join(self.data_dir, 'Images-512x512')
    
    if split == 'test':
      self.data_dir = os.path.join(self.data_dir, 'VisDrone2018-DET-val')
      self.img_dir = os.path.join(self.data_dir, 'images')
      self.annot_dir = os.path.join(self.data_dir, 'annotations')
      self.annot_path = os.path.join(
          self.annot_dir, 
          'instances.json').format(split)
          
    else:
      self.data_dir = os.path.join(self.data_dir, 'VisDrone2018-DET-train')
      self.img_dir = os.path.join(self.data_dir, 'images')
      self.annot_dir = os.path.join(self.data_dir, 'annotations')
      self.annot_path = os.path.join(
        self.annot_dir, 
          'instances.json').format(split)

    self.max_objs = 128
    self.class_name = [
    'Ignored Regions', 'Pedestrian', 'People', 'Bicycle', 
    'Car', 'Van', 'Truck', 'Tricycle', 'Awning-tricycle', 
    'Bus', 'Motorbike', 'Other']

    self._valid_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}

    self.voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) \
                      for v in range(1, self.num_classes + 1)]
    self._data_rng = np.random.RandomState(123)
    self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                             dtype=np.float32)
    self._eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32)
    self.mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
    self.std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

    self.split = split
    self.opt = opt

    print('==> initializing visdrone {} data.'.format(split))
    self.coco = coco.COCO(self.annot_path)
    self.images = self.coco.getImgIds()
    self.num_samples = len(self.images)

    print('Loaded {} {} samples'.format(split, self.num_samples))

  def _to_float(self, x):
    return float("{:.2f}".format(x))

  def convert_eval_format(self, all_bboxes):
    # import pdb; pdb.set_trace()
    detections = []
    for image_id in all_bboxes:
      for cls_ind in all_bboxes[image_id]:
        category_id = self._valid_ids[cls_ind - 1]
        for bbox in all_bboxes[image_id][cls_ind]:
          bbox[2] -= bbox[0]
          bbox[3] -= bbox[1]
          score = bbox[4]
          bbox_out  = list(map(self._to_float, bbox[0:4]))

          detection = {
              "image_id": int(image_id),
              "category_id": int(category_id),
              "bbox": bbox_out,
              "score": float("{:.2f}".format(score))
          }
          if len(bbox) > 5:
              extreme_points = list(map(self._to_float, bbox[5:13]))
              detection["extreme_points"] = extreme_points
          detections.append(detection)
    return detections

  def __len__(self):
    return self.num_samples

  def save_results(self, results, save_dir):
    json.dump(self.convert_eval_format(results), 
                open('{}/results.json'.format(save_dir), 'w'))
  
  def run_eval(self, results, save_dir):
    # result_json = os.path.join(save_dir, "results.json")
    # detections  = self.convert_eval_format(results)
    # json.dump(detections, open(result_json, "w"))
    self.save_results(results, save_dir)
    coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
    coco_eval = COCOeval(self.coco, coco_dets, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()




# class ImageFolder(data.Dataset):
#     def __init__(self, folder_path, img_size=416):
#         self.files = sorted(glob.glob('%s/*.*' % folder_path))
#         self.img_shape = (img_size, img_size)

#     def __getitem__(self, index):
#         img_path = self.files[index % len(self.files)]
#         # Extract image
#         img = np.array(Image.open(img_path))
#         h, w, _ = img.shape
#         dim_diff = np.abs(h - w)
#         # Upper (left) and lower (right) padding
#         pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
#         # Determine padding
#         pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
#         # Add padding
#         input_img = np.pad(img, pad, 'constant', constant_values=127.5) / 255.
#         # Resize and normalize
#         input_img = resize(input_img, (*self.img_shape, 3), mode='reflect')
#         # Channels-first
#         input_img = np.transpose(input_img, (2, 0, 1))
#         # As pytorch tensor
#         input_img = torch.from_numpy(input_img).float()

#         return img_path, input_img

#     def __len__(self):
#         return len(self.files)


# class VisdroneDetection(data.Dataset):
#     def __init__(self, list_path, img_size=416):
#         with open(list_path, 'r') as file:
#             self.img_files = file.readlines()
#         self.label_files = [path.replace('images', 'annotations').replace('.png', '.txt').replace('.jpg', '.txt') for path in self.img_files]
#         self.img_shape = (img_size, img_size)
#         self.max_objects = 50

#     def __getitem__(self, index):

#         #---------
#         #  Image
#         #---------

#         img_path = self.img_files[index % len(self.img_files)].rstrip()
#         img = np.array(Image.open(img_path))

#         # Handles images with less than three channels
#         while len(img.shape) != 3:
#             index += 1
#             img_path = self.img_files[index % len(self.img_files)].rstrip()
#             img = np.array(Image.open(img_path))

#         h, w, _ = img.shape
#         dim_diff = np.abs(h - w)
#         # Upper (left) and lower (right) padding
#         pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
#         # Determine padding
#         pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
#         # Add padding
#         input_img = np.pad(img, pad, 'constant', constant_values=128) / 255.
#         padded_h, padded_w, _ = input_img.shape
#         # Resize and normalize
#         input_img = resize(input_img, (*self.img_shape, 3), mode='reflect', anti_aliasing=False)
#         # Channels-first
#         input_img = np.transpose(input_img, (2, 0, 1))
#         # As pytorch tensor
#         input_img = torch.from_numpy(input_img).float()

#         #---------
#         #  Label
#         #---------

#         label_path = self.label_files[index % len(self.img_files)].rstrip()

#         labels = None
#         if os.path.exists(label_path):
#             #EDIT FOR VISDRONE
#             labels = np.loadtxt(label_path, delimiter=',', usecols=range(8)).reshape(-1,8)
#             labels[:,1:5] = labels[:,0:4]
#             labels[:,0] = labels[:,5]
#             labels = labels[:,:5]
#             labels = labels[labels[:,0]!=0,:]#remove 0 category
#             labels[:,0] = labels[:,0]-1
#             # Extract coordinates for unpadded + unscaled image
#             x1 = (labels[:, 1] - labels[:, 3]/2)
#             y1 = (labels[:, 2] - labels[:, 4]/2)
#             x2 = (labels[:, 1] + labels[:, 3]/2)
#             y2 = (labels[:, 2] + labels[:, 4]/2)

#             # Adjust for added padding
#             x1 += pad[1][0]
#             y1 += pad[0][0]
#             x2 += pad[1][0]
#             y2 += pad[0][0]
#             # Calculate ratios from coordinates
#             labels[:, 1] = ((x1 + x2) / 2) / padded_w
#             labels[:, 2] = ((y1 + y2) / 2) / padded_h
#             labels[:, 3] /= padded_w
#             labels[:, 4] /= padded_h

#         boxes = torch.zeros((len(labels),6))
#         boxes[:, 1:] = torch.from_numpy(labels)

#         return img_path, input_img.float(), boxes.float()

    

#     @staticmethod
#     def collate_fn(batch):
#         paths, imgs, labels = list(zip(*batch))
#         for i, boxes in enumerate(labels):
#             boxes[:, 0] = i
#         imgs = torch.stack(imgs,0)
#         labels = torch.cat(labels, 0)
#         return paths, imgs, labels

#     def __len__(self):
#         return len(self.img_files)
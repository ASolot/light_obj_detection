import numpy as np
import glob
import cv2
import os
from tqdm import tqdm

X = 1
Y = 2
W = 3
H = 4

Xc = 1
Yc = 2

path_train_imgs = "VisDrone2018-DET-train/images/"
path_eval_imgs  = "VisDrone2018-DET-val/images/"

path_train_labels = "VisDrone2018-DET-train/annotations/*.txt"
path_eval_labels  = "VisDrone2018-DET-val/annotations/*.txt"

def load_annotations(path):
    return np.reshape(np.loadtxt(path, dtype=np.float32, delimiter=",", usecols=(0,1,2,3,5)), (-1, 5))

def filename_from_labelname(path, root_path):
    filename = path.rsplit('/', 1)[1]
    imgpath = root_path + filename.rsplit('.',1)[0] + ".jpg"
    return imgpath

def process_labels(generic_label_path, generic_image_root_path, message): 

    # generating the list of paths to be processed
    path_list = sorted(glob.glob(generic_label_path))

    # for each path, convert the labels
    for path in tqdm(path_list, desc=message):
        
        annotations = load_annotations(path)
        inp = cv2.imread(filename_from_labelname(path, generic_image_root_path))

        for j in range(annotations.shape[0]):
            tmp = annotations[j]
            tmp = np.insert(tmp, 0, tmp[-1])
            tmp = np.delete(tmp, -1)
            annotations[j] = tmp

            # moving from corner coordinates to center coordinates
            annotations[j][Xc] = annotations[j][X] + annotations[j][W]/2.0
            annotations[j][Yc] = annotations[j][Y] + annotations[j][H]/2.0 

            # scaling to 0 - 1
            annotations[j][Xc] /= inp.shape[1] #returns the no of cols, thus width
            annotations[j][Yc] /= inp.shape[0] #returns the no of rows, thus height 
            annotations[j][W]  /= inp.shape[1]
            annotations[j][H]  /= inp.shape[0]


        np.savetxt(path+"_transformed", annotations, fmt="%1.0f %.9f %.9f %.9f %.9f", delimiter=" ")
    return

process_labels(path_train_labels, path_train_imgs, "Converting train labels")
process_labels(path_eval_labels, path_eval_imgs, "Converting eval labels")

print ("Done")




import gzip
import numpy as np
import cv2
import sys
import glob
import random
import os
from PIL import Image

#DATADIR = '/harddisk/hdd_c/camelyon/code1/2015dataset-HSV' #data_dirs.mnist
DATADIR = '/harddisk/hdd_c/camelyon/code1/2015dataset' #data_dirs.mnist
LABELS = ['Normal','Benign', 'InSitu', 'Invasive']
LABELS_2 = ['non-carcinoma', 'carcinoma']

NUM_LABELS = 4
NUM_LABELS_2 = 2

IMAGE_SHAPE = [2048, 1536, 3]
PATCH_SHAPE = [256, 256, 3]
OVERLOP = 128

def get_filename_label(name):
  # load images' fliename
  if name is 'Train':
    train_filename = []
    train_label = []
    val_filename = []
    val_label = []
    for index in range(len(LABELS)):
      image_path_names = glob.glob(DATADIR + '/' + 'Training_data' + '/' + LABELS[index] + '/*.tif')
      if LABELS[index] == 'Normal' or LABELS[index] == 'Benign':
        label_1 = [0] * len(image_path_names)
      else:
        label_1 = [1] * len(image_path_names)
      
      train_filename.extend(image_path_names[0:int(len(image_path_names)*0.9)])
      train_label.extend(label_1[0:int(len(image_path_names)*0.9)])
      val_filename.extend(image_path_names[int(len(image_path_names)*0.9):])
      val_label.extend(label_1[int(len(image_path_names)*0.9):])
    return train_filename, train_label, val_filename, val_label
  elif name is 'Test':
    test_fliename = []
    test_label = []
    labels_file = glob.glob(DATADIR + '/' + 'Test_data' + '/labels.txt')
    fp = open(labels_file[0],'r')
    lines = fp.readlines()
    fp.close()
    labels = []
    for line in lines:
      test_fliename.append(line.split('\t')[0])
      labels.append(line.split('\t')[1])
    for i in range(len(test_fliename)):
      if labels[i] == 'Normal' or labels[i] == 'Benign':
        test_label.append(0)
      else:
        test_label.append(1)
    return test_fliename, test_label



def get_unsup_sup_filename(train_filename, train_label, sup_per_class):
  t_sup_filename = []
  t_sup_labels = []
  t_unsup_filename = []
  for i in range(len(LABELS)):
    a = []
    for j in range(len(train_label)):
      if (train_label[j] == i) is True:
         a.append(train_filename[j])
        
    if sup_per_class == -1:  # use all available labeled data
      t_sup_filename.extend(a)
      t_sup_labels.extend([i] * len(a))
      t_unsup_filename = t_sup_filename
    else:  # use randomly chosen subset
      t_unsup_filename.extend(a)
      np.random.seed(11)#1
      np.random.shuffle(a)
      t_sup_filename.extend(a[0:sup_per_class])
      t_sup_labels.extend([i] * sup_per_class)
#       t_unsup_filename.extend(a)
  return t_unsup_filename, t_sup_filename, t_sup_labels
    


def get_image_patch(filename, label):
  patches = []
  labels = []
  if label is None:
    for i in range(len(filename)):
      img = Image.open(filename[i])
      patch = extract_patches(img, PATCH_SHAPE[0], OVERLOP)
      patches.extend(patch)
  else:
    for i in range(len(filename)):
      img = Image.open(filename[i])
      patch = extract_patches(img, PATCH_SHAPE[0], OVERLOP)
      patches.extend(patch)
      labels.extend([label[i]] * len(patch))
  return np.array(patches, dtype = np.float32), np.array(labels, dtype = np.int32)

    
def extract_patches(img, stride, overlop):
  index_h = 1
  i = 0
  patches = []
  while i + stride <= IMAGE_SHAPE[0]:
    index_w = 1
    j = 0
    while j + stride <= IMAGE_SHAPE[1]:
      patch=img.crop((i, #left
                      j, #up
                      i + stride, #right
                      j + stride #down
                      ))
      j = j + overlop
      patches.append(np.asarray(patch))
      index_w = index_w + 1
    index_h = index_h + 1
    i = i + overlop
  return patches





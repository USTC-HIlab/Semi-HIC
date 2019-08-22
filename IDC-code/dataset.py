import gzip
import numpy as np
import cv2
import sys
import glob
import random
import os
from PIL import Image

DATADIR = '/harddisk/hdd_c/camelyon/code1/2015dataset-HSV' #data_dirs.mnist
LABELS = ['0','1']


NUM_LABELS = 2

IMAGE_SHAPE = [50, 50, 3]



def get_data():
  images_0 = np.load('/harddisk/hdd_c/camelyon/code1/IDC_dataset_hsv/0.npy')
  labels_0 = np.array([0]*len(images_0))
  images_1 = np.load('/harddisk/hdd_c/camelyon/code1/IDC_dataset_hsv/1.npy')
  labels_1 = np.array([1]*len(images_1))
    
  #shuffer
  np.random.seed(1234)
  
  num_example = images_0.shape[0]
  arr = np.arange(int(num_example))
  np.random.shuffle(arr)
  images_0 = images_0[arr]
  labels_0 = labels_0[arr]
  
  num_example = images_1.shape[0]
  arr = np.arange(int(num_example))
  np.random.shuffle(arr)
  images_1 = images_1[arr]
  labels_1 = labels_1[arr]
    
  # train,val,test
  train_images_0 = images_0[0:int(len(images_0)*0.6)]
  train_labels_0 = labels_0[0:int(len(labels_0)*0.6)]
  val_images_0 = images_0[int(len(images_0)*0.6):int(len(images_0)*0.8)]
  val_labels_0 = labels_0[int(len(labels_0)*0.6):int(len(labels_0)*0.8)]
  test_images_0 = images_0[int(len(images_0)*0.8):]
  test_labels_0 = labels_0[int(len(labels_0)*0.8):]
  
  train_images_1 = images_1[0:int(len(images_1)*0.6)]
  train_labels_1 = labels_1[0:int(len(labels_1)*0.6)]
  val_images_1 = images_1[int(len(images_1)*0.6):int(len(images_1)*0.8)]
  val_labels_1 = labels_1[int(len(labels_1)*0.6):int(len(labels_1)*0.8)]
  test_images_1 = images_1[int(len(images_1)*0.8):]
  test_labels_1 = labels_1[int(len(labels_1)*0.8):]
  
  # down_sampling
  num_example = train_images_0.shape[0]
  arr = np.arange(int(num_example))
  np.random.shuffle(arr)
  train_images_0 = train_images_0[arr[0:len(train_images_1)]]
  train_labels_0 = train_labels_0[arr[0:len(train_images_1)]]

  # train
  train_images = np.vstack((train_images_0, train_images_1))
  train_labels = np.hstack((train_labels_0, train_labels_1))
  
  # validation
  val_images = np.vstack((val_images_0, val_images_1))
  val_labels = np.hstack((val_labels_0, val_labels_1))


  # test
  test_images = np.vstack((test_images_0, test_images_1))
  test_labels = np.hstack((test_labels_0, test_labels_1))
  return train_images, train_labels, val_images, val_labels, test_images, test_labels

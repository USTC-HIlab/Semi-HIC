import tensorflow as tf
from backend import * 
import architectures
import sys
import numpy as np
import tensorflow.contrib.slim as slim

from tensorflow.python.platform import app
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS


import dataset as dataset_tools 
import sys
NUM_LABELS = dataset_tools.NUM_LABELS
IMAGE_SHAPE = dataset_tools.IMAGE_SHAPE

def main(_):
  train_images, train_labels, val_images, val_labels, test_images, test_labels = dataset_tools.get_data()
  
  graph = tf.Graph()
  
  with graph.as_default():

    model = SemisupModel(architectures.dataset_model, NUM_LABELS, IMAGE_SHAPE)
    saver = tf.train.Saver()
  with tf.Session(graph=graph) as sess:   
    ckpt = '/harddisk/hdd_c/camelyon/code1/new-2015-test/IDC-new/model/model-all-3000/model-11000'
#     ckpt = '/harddisk/hdd_c/camelyon/code1/new-2015-test/IDC-new/model/model-30000-3000/model-17000'
#     ckpt = '/harddisk/hdd_c/camelyon/code1/new-2015-test/IDC-new/model/model-2000-3000/model-18000'
#     ckpt = '/harddisk/hdd_c/camelyon/code1/new-2015-test/IDC-new/model/model-100-1000/model-9500'
    
    saver.restore(sess, ckpt)
    val_pred = model.classify(val_images).argmax(-1)
    conf_mtx = confusion_matrix(val_labels, val_pred, NUM_LABELS)
    val_err = (val_labels != val_pred).mean() * 100
    print(conf_mtx)
    print('Validation error: %.2f %%' % val_err)
    print()
    
    # test
    test_pred = model.classify(test_images).argmax(-1)
    
    output = model.classify(test_images)
    
    conf_mtx = confusion_matrix(test_labels, test_pred, NUM_LABELS)
    test_err = (test_labels != test_pred).mean() * 100
    print(conf_mtx)
    print('Test error: %.2f %%' % test_err)
    print()
    

if __name__ == '__main__':
  app.run()
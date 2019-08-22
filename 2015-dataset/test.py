import tensorflow as tf
from backend import * 
import architectures
import sys
import numpy as np
import tensorflow.contrib.slim as slim

from tensorflow.python.platform import app
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

flags.DEFINE_integer('sup_per_class', -1,
                     'Number of labeled samples used per class.-1 = all')



import dataset as dataset_tools
NUM_LABELS = dataset_tools.NUM_LABELS_2
IMAGE_SHAPE = dataset_tools.IMAGE_SHAPE
PATCH_SHAPE = dataset_tools.PATCH_SHAPE

def main(_):
  train_filename, train_label, val_filename, val_label = dataset_tools.get_filename_label('Train')
  test_filename, test_labels = dataset_tools.get_filename_label('Test')
  t_unsup_filename, t_sup_filename, t_sup_labels = dataset_tools.get_unsup_sup_filename(train_filename, train_label, FLAGS.sup_per_class)
  #train
  t_unsup_patch, _ = dataset_tools.get_image_patch(t_unsup_filename, None)
  t_sup_patch, t_sup_patch_labels = dataset_tools.get_image_patch(t_sup_filename, t_sup_labels)

  t1_val_patch, t1_val_patch_labels = dataset_tools.get_image_patch(val_filename, val_label) #validation
  t1_test_patch, t1_test_patch_labels = dataset_tools.get_image_patch(test_filename, test_labels) #test

  graph = tf.Graph()
  
  with graph.as_default():

    model = SemisupModel(architectures.dataset_model, NUM_LABELS, PATCH_SHAPE)
    saver = tf.train.Saver()
  with tf.Session(graph=graph) as sess:   
    ckpt = '/harddisk/hdd_c/camelyon/code1/new-2015-test/class2-new/model/model-all-3000/model-7499'
#     ckpt = '/harddisk/hdd_c/camelyon/code1/new-2015-test/class2-new/model/model-80-3000/model-16999'
#     ckpt = '/harddisk/hdd_c/camelyon/code1/new-2015-test/class2-new/model/model-40-3000/model-3999'
#     ckpt = '/harddisk/hdd_c/camelyon/code1/new-2015-test/class2-new/model/model-20-3000/model-9999'

    saver.restore(sess, ckpt)
    
    patch_test_pred = []
    image_test_pred_max = []
    image_test_pred_sum = []
    image_test_pred_maj = []
    t_test_patch_labels = []
    print('Test:')
    for i in range(len(test_filename)):
      t_test_patch, t_test_patch_labels_1 = dataset_tools.get_image_patch([test_filename[i]], [test_labels[i]]) 
      t_test_patch_labels.extend(t_test_patch_labels_1)
      
      ## max
      patch_test_pred_1 = model.classify(t_test_patch).argmax(-1)
      patch_test_pred_2 = model.classify(t_test_patch)
      index_1_max = model.classify(t_test_patch).argmax(0)
      image_test_pred_max.append( np.argmax([patch_test_pred_2[index_1_max[0],0],patch_test_pred_2[index_1_max[1],1]]) )
      patch_test_pred.extend(patch_test_pred_1)
      
      ## sum
      index_1_sum = sum(model.classify(t_test_patch))
      image_test_pred_sum.append( np.argmax(index_1_sum) )

      
      ## maj
      image_test_pred_maj.extend([max(set(list(patch_test_pred_1)), key=list(patch_test_pred_1).count)])
      
    ##vote
    image_test_pred_vote = 

    
    print('Patch:')
    conf_mtx_patch = confusion_matrix(np.array(t_test_patch_labels), np.array(patch_test_pred), NUM_LABELS)
    print(conf_mtx_patch)
    test_err_patch = (np.array(t_test_patch_labels) != np.array(patch_test_pred)).mean() * 100
    print('Test patch error: %.2f %%' % test_err_patch)
    print()
        
    print('Image:')
    conf_mtx_image = confusion_matrix(np.array(test_labels), np.array(image_test_pred), NUM_LABELS)
    print(conf_mtx_image)
    test_err_image = (np.array(test_labels) != np.array(image_test_pred)).mean() * 100
    print('Test image error: %.2f %%' % test_err_image)
    print()
    
    print(test_labels)
    print(image_test_pred)
if __name__ == '__main__':
  app.run()    
 
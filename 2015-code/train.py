from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from backend import * 
import architectures
import sys
import numpy as np

from tensorflow.python.platform import app
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

flags.DEFINE_integer('sup_per_class', -1,
                     'Number of labeled samples used per class.-1 = all')

flags.DEFINE_integer('sup_per_batch', 32,#32
                     'Number of labeled samples per class per batch.')

flags.DEFINE_integer('unsup_batch_size', 64,#64
                     'Number of unlabeled samples per batch.')

flags.DEFINE_integer('eval_interval', 500,
                     'Number of steps between evaluations.')

flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')

flags.DEFINE_float('decay_factor', 0.33, 'Learning rate decay factor.')

flags.DEFINE_float('decay_steps', 3000,
                   'Learning rate decay interval in steps.')

flags.DEFINE_float('visit_weight', 0.75, 'Weight for visit loss.')

flags.DEFINE_float('walker_weight', 1.0, 'Weight for walker loss.')

flags.DEFINE_integer('max_steps', 20000, 'Number of training steps.')

flags.DEFINE_string('checkpoint_dir', '/harddisk/hdd_c/camelyon/code1/new-2015-test/class2-new/result/model-all-3000/model', 
                    'Save checkpoint path.')

flags.DEFINE_string('logdir', '/harddisk/hdd_c/camelyon/code1/new-2015-test/class2-new/semisup_bach/semi-all-3000', 'Training log path.')


import dataset as dataset_tools
NUM_LABELS = dataset_tools.NUM_LABELS_2
IMAGE_SHAPE = dataset_tools.IMAGE_SHAPE
PATCH_SHAPE = dataset_tools.PATCH_SHAPE

def main(_):
  train_filename, train_label, val_filename, val_label = dataset_tools.get_filename_label('Train')
  test_filename, test_labels = dataset_tools.get_filename_label('Test')
  t_unsup_filename, t_sup_filename, t_sup_labels = dataset_tools.get_unsup_sup_filename(train_filename, train_label, FLAGS.sup_per_class)
  
  #train, validation, test
  t_unsup_patch, _ = dataset_tools.get_image_patch(t_unsup_filename, None)
  t_sup_patch, t_sup_patch_labels = dataset_tools.get_image_patch(t_sup_filename, t_sup_labels)

  t1_val_patch, t1_val_patch_labels = dataset_tools.get_image_patch(val_filename, val_label) #validation



  num_example = t_unsup_patch.shape[0]
  arr = np.arange(int(num_example))
  np.random.seed(1)
  np.random.shuffle(arr)
  t_unsup_patch = t_unsup_patch[arr]
    
  num_example = t_sup_patch.shape[0]
  arr = np.arange(int(num_example))
  np.random.shuffle(arr)
  t_sup_patch = t_sup_patch[arr]
  t_sup_patch_labels = t_sup_patch_labels[arr]
   
  graph = tf.Graph()
  with graph.as_default():
    model = SemisupModel(architectures.dataset_model, NUM_LABELS, PATCH_SHAPE)
    
    t_unsup_images = tf.placeholder(tf.float32, [FLAGS.unsup_batch_size] + PATCH_SHAPE, name='t_unsup_images')
    t_sup_images = tf.placeholder(tf.float32, [FLAGS.sup_per_batch] + PATCH_SHAPE, name='t_sup_images')
    t_sup_labels = tf.placeholder(tf.int32, [FLAGS.sup_per_batch] + [], name='t_sup_labels')
    
   
    # Compute embeddings and logits.
    t_sup_emb = model.image_to_embedding(t_sup_images)
    t_unsup_emb = model.image_to_embedding(t_unsup_images)
    t_sup_logit = model.embedding_to_logit(t_sup_emb)
    
    # Add losses.
    model.add_semisup_loss(t_sup_emb, t_unsup_emb, t_sup_labels, walker_weight=FLAGS.walker_weight, visit_weight=FLAGS.visit_weight)
    model.add_logit_loss(t_sup_logit, t_sup_labels)

    t_learning_rate = tf.train.exponential_decay(
        FLAGS.learning_rate,
        model.step,
        FLAGS.decay_steps,
        FLAGS.decay_factor,
        staircase=True)
    train_op, train_loss = model.create_train_op(t_learning_rate)
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(FLAGS.logdir, graph)

    saver = tf.train.Saver()
    
  with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()
    
    
    for step in xrange(FLAGS.max_steps):
      unsup_batch = next_batch(t_unsup_patch, [],FLAGS.unsup_batch_size, step, 'unsup_batch')
      sup_batch = next_batch(t_sup_patch, t_sup_patch_labels, FLAGS.sup_per_batch, step, 'sup_batch')
      _, loss, summaries = sess.run([train_op, train_loss, summary_op],feed_dict={t_unsup_images: unsup_batch, 
                                                                            t_sup_images: sup_batch[0], 
                                                                            t_sup_labels: sup_batch[1]})
              
      if step % 10 == 0:
        Validation_loss = model.classify_loss(t1_val_patch, t1_val_patch_labels)
        Validation_loss_summary = tf.Summary(
            value=[tf.Summary.Value(
                tag='Validation Loss', simple_value=Validation_loss)])
        
        summary_writer.add_summary(summaries, step)
        summary_writer.add_summary(Validation_loss_summary, step)
      
        
        
        
        patch_val_pred_2 = model.classify(t1_val_patch).argmax(-1)
        Validation_acc = 100 - (np.array(t1_val_patch_labels) != np.array(patch_val_pred_2)).mean() * 100
        
        Validation_acc_summary = tf.Summary(
            value=[tf.Summary.Value(
                tag='Test acc', simple_value=Validation_acc)])
        summary_writer.add_summary(Validation_acc_summary, step)
        
        
        
        
        
      if (step + 1) % FLAGS.eval_interval == 0 or step == 99:
        print('Training loss: %f' % loss)        
        
        test_loss = model.classify_loss(t1_val_patch, t1_val_patch_labels)
        print('Validation loss: %f' % Validation_loss)

        
        #validation        
        patch_val_pred = []
        image_val_pred = []
        t_val_patch_labels = []
        print('Step: %d' % step)
        print('Validation:')
        for i in range(len(val_filename)):
          t_val_patch, t_val_patch_labels_1 = dataset_tools.get_image_patch([val_filename[i]], [val_label[i]]) 
          t_val_patch_labels.extend(t_val_patch_labels_1)
          patch_val_pred_1 = model.classify(t_val_patch).argmax(-1)
          image_val_pred.extend([max(set(list(patch_val_pred_1)), key=list(patch_val_pred_1).count)])
          patch_val_pred.extend(patch_val_pred_1)
        
        
        
        print('Patch:')
        conf_mtx_patch = confusion_matrix(np.array(t_val_patch_labels), np.array(patch_val_pred), NUM_LABELS)
        print(conf_mtx_patch)
        val_err_patch = (np.array(t_val_patch_labels) != np.array(patch_val_pred)).mean() * 100
        print('Validation patch error: %.2f %%' % val_err_patch)
        print()
        
        print('Image:')
        conf_mtx_image = confusion_matrix(np.array(val_label), np.array(image_val_pred), NUM_LABELS)
        print(conf_mtx_image)
        val_err_image = (np.array(val_label) != np.array(image_val_pred)).mean() * 100
        print('Validation image error: %.2f %%' % val_err_image)
        print()
        
        saver.save(sess, FLAGS.checkpoint_dir, global_step=step)






if __name__ == '__main__':
  app.run()

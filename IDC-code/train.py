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
                     'Number of labeled samples used per class.')

flags.DEFINE_integer('sup_seed', -1,
                     'Integer random seed used for labeled set selection.')

flags.DEFINE_integer('sup_per_batch', 16,
                     'Number of labeled samples per class per batch.')

flags.DEFINE_integer('unsup_batch_size', 64,
                     'Number of unlabeled samples per batch.')

flags.DEFINE_integer('eval_interval', 500,
                     'Number of steps between evaluations.')

flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')

flags.DEFINE_float('decay_factor', 0.33, 'Learning rate decay factor.')

flags.DEFINE_float('decay_steps', 4000,
                   'Learning rate decay interval in steps.')

flags.DEFINE_float('visit_weight', 0.65, 'Weight for visit loss.')

flags.DEFINE_integer('max_steps', 20000, 'Number of training steps.')

flags.DEFINE_string('checkpoint_dir', '/harddisk/hdd_c/camelyon/code1/new-2015-test/IDC-new/result/model-all-3000-all/model', 
                    'Save checkpoint path.')

flags.DEFINE_string('logdir', '/harddisk/hdd_c/camelyon/code1/new-2015-test/IDC-new/semisup_bach/semi-all-3000-all', 'Training log path.')

import dataset as dataset_tools 
import sys
NUM_LABELS = dataset_tools.NUM_LABELS
IMAGE_SHAPE = dataset_tools.IMAGE_SHAPE


def main(_):
  train_images, train_labels, val_images, val_labels, test_images, test_labels = dataset_tools.get_data()


  # Sample labeled training subset.
  seed = FLAGS.sup_seed if FLAGS.sup_seed != -1 else None
  sup_by_label = sample_by_label(train_images, train_labels,
                                         FLAGS.sup_per_class, NUM_LABELS, seed)

  graph = tf.Graph()
  with graph.as_default():
    model = SemisupModel(architectures.dataset_model, NUM_LABELS, IMAGE_SHAPE)
    
#     unsup_num = 3000
    # Set up inputs.
#     t_unsup_images, _ = create_input(train_images[0:unsup_num], train_labels[0:unsup_num], FLAGS.unsup_batch_size)
    t_unsup_images, _ = create_input(train_images, train_labels, FLAGS.unsup_batch_size)
       
    t_sup_images, t_sup_labels = create_per_class_inputs(sup_by_label, FLAGS.sup_per_batch)

    # Compute embeddings and logits.
    t_sup_emb = model.image_to_embedding(t_sup_images)
    t_unsup_emb = model.image_to_embedding(t_unsup_images)
    t_sup_logit = model.embedding_to_logit(t_sup_emb)

    # Add losses.
    model.add_semisup_loss(t_sup_emb, t_unsup_emb, t_sup_labels, visit_weight = FLAGS.visit_weight)
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

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for step in xrange(FLAGS.max_steps):
      _, loss ,summaries = sess.run([train_op, train_loss, summary_op])

    
      if step % 10 == 0:
        test_loss = model.classify_loss(val_images, val_labels)
#         print(test_loss)
        test_loss_summary = tf.Summary(
            value=[tf.Summary.Value(
                tag='Validation Loss', simple_value=test_loss)])
        
        summary_writer.add_summary(summaries, step)
        summary_writer.add_summary(test_loss_summary, step)
             
        val_pred_2 = model.classify(val_images).argmax(-1)
        test_acc = 100 - (np.array(val_labels) != np.array(val_pred_2)).mean() * 100
        
        test_acc_summary = tf.Summary(
            value=[tf.Summary.Value(
                tag='Validation acc', simple_value=test_acc)])
        summary_writer.add_summary(test_acc_summary, step)

      
      if (step + 1) % FLAGS.eval_interval == 0 or step == 99:
        print('Step: %d' % step)
        
        # validation
        val_pred = model.classify(val_images).argmax(-1)
        conf_mtx = confusion_matrix(val_labels, val_pred, NUM_LABELS)
        val_err = (val_labels != val_pred).mean() * 100
        print(conf_mtx)
        print('Validation error: %.2f %%' % val_err)
        print()


        saver.save(sess, FLAGS.checkpoint_dir, model.step)

    coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':
  app.run()

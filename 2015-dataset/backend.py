from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import sys
import tensorflow as tf
import tensorflow.contrib.slim as slim

def next_batch(images, labels, batch_size, step, name):
  num_examples = len(images)

  #shuffer
  if step % (int(num_examples/batch_size)) == 0:
#     if name is 'sup_batch':
#       training_data = np.array(zip(images, labels))
#       np.random.shuffle(training_data)
#       images = np.array([x[0] for x in training_data])
#       labels = training_data[:,1]
#     else:
#       np.random.shuffle(images)
#     training_data = []
    
    
    num_example = images.shape[0]
    arr = np.arange(int(num_example))
    np.random.shuffle(arr)
    images = images[arr]
    if name is 'sup_batch':
      labels = labels[arr]
    
  
  start = (step * batch_size) % num_examples
  index_in_epoch = 0
  if name is 'sup_batch':
    if start + batch_size > num_examples:
      rest_num_examples = num_examples - start
      images_rest_part = images[start:num_examples]
      labels_rest_part = labels[start:num_examples]
      start = 0
      index_in_epoch = batch_size - rest_num_examples
      end = index_in_epoch
      images_new_part = images[start:end] 
      labels_new_part = labels[start:end]
      return np.concatenate((images_rest_part, images_new_part), axis=0) ,np.concatenate((labels_rest_part, labels_new_part), axis=0)
    else:
      index_in_epoch = batch_size + start 
      end = index_in_epoch
      return images[start:end], labels[start:end]
  elif name is 'unsup_batch':
    if start + batch_size > num_examples:
      rest_num_examples = num_examples - start
      images_rest_part = images[start:num_examples]
      start = 0
      index_in_epoch = batch_size - rest_num_examples
      end = index_in_epoch
      images_new_part = images[start:end] 
      return np.concatenate((images_rest_part, images_new_part), axis=0)
    else:
      index_in_epoch = batch_size + start 
      end = index_in_epoch
      return images[start:end]


def create_virt_emb(n, size):
  """Create virtual embeddings."""
  emb = slim.variables.model_variable(
      name='virt_emb',
      shape=[n, size],
      dtype=tf.float32,
      trainable=True,
      initializer=tf.random_normal_initializer(stddev=0.01))
  return emb


def confusion_matrix(labels, predictions, num_labels):
  """Compute the confusion matrix."""
  rows = []
  for i in xrange(num_labels):
    row = np.bincount(predictions[labels == i], minlength=num_labels)
    rows.append(row)
  return np.vstack(rows)



class SemisupModel(object):
  """Helper class for setting up semi-supervised training."""

  def __init__(self, model_func, num_labels, input_shape, test_in=None):
    """Initialize SemisupModel class.

    Creates an evaluation graph for the provided model_func.

    Args:
      model_func: Model function. It should receive a tensor of images as
          the first argument, along with the 'is_training' flag.
      num_labels: Number of taget classes.
      input_shape: List, containing input images shape in form
          [height, width, channel_num].
      test_in: None or a tensor holding test images. If None, a placeholder will
        be created.
    """

    self.num_labels = num_labels
    self.step = slim.get_or_create_global_step()
    self.ema = tf.train.ExponentialMovingAverage(0.99, self.step)

    self.test_batch_size = 100

    self.model_func = model_func

    if test_in is not None:
      self.test_in = test_in
      self.test_label = test_label
    else:
      self.test_in = tf.placeholder(np.float32, [None] + input_shape, 'test_in')
      self.test_label = tf.placeholder(np.int32, [None] + [], 'test_label')

    self.test_emb = self.image_to_embedding(self.test_in, is_training=False)
    self.test_logit = self.embedding_to_logit(self.test_emb, is_training=False)
    self.test_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf.one_hot(self.test_label, 
                                                                                            self.test_logit.get_shape()[-1]), 
                                                                            logits = self.test_logit))

    
  def image_to_embedding(self, images, is_training=True):
    """Create a graph, transforming images into embedding vectors."""
    with tf.variable_scope('net', reuse=is_training):
      return self.model_func(images, is_training=is_training)

  def embedding_to_logit(self, embedding, is_training=True):
    """Create a graph, transforming embedding vectors to logit classs scores."""
    with tf.variable_scope('net', reuse=is_training):
      return slim.fully_connected(
          embedding,
          self.num_labels,
          activation_fn=None,
          normalizer_fn=slim.batch_norm,
          normalizer_params={'is_training': is_training},
          weights_regularizer=slim.l2_regularizer(1e-4))

  def add_semisup_loss(self, a, b, labels, walker_weight=1.0, visit_weight=1.0):
    """Add semi-supervised classification loss to the model.

    The loss constist of two terms: "walker" and "visit".

    Args:
      a: [N, emb_size] tensor with supervised embedding vectors.
      b: [M, emb_size] tensor with unsupervised embedding vectors.
      labels : [N] tensor with labels for supervised embeddings.
      walker_weight: Weight coefficient of the "walker" loss.
      visit_weight: Weight coefficient of the "visit" loss.
    """

    equality_matrix = tf.equal(tf.reshape(labels, [-1, 1]), labels)
    equality_matrix = tf.cast(equality_matrix, tf.float32)
    p_target = (equality_matrix / tf.reduce_sum(
        equality_matrix, [1], keep_dims=True))

    match_ab = tf.matmul(a, b, transpose_b=True, name='match_ab')
    p_ab = tf.nn.softmax(match_ab, name='p_ab')
    p_ba = tf.nn.softmax(tf.transpose(match_ab), name='p_ba')
    p_aba = tf.matmul(p_ab, p_ba, name='p_aba')

    self.create_walk_statistics(p_aba, equality_matrix)
    
    loss_aba = tf.losses.softmax_cross_entropy(
    #tf.losses.sigmoid_cross_entropy(
    #tf.losses.softmax_cross_entropy(
        p_target,
        tf.log(1e-8 + p_aba),
        weights=walker_weight,
        scope='loss_aba')
    self.add_visit_loss(p_ab, visit_weight)

    tf.summary.scalar('Loss_aba', loss_aba)

  def add_visit_loss(self, p, weight=1.0):
    """Add the "visit" loss to the model.

    Args:
      p: [N, M] tensor. Each row must be a valid probability distribution
          (i.e. sum to 1.0)
      weight: Loss weight.
    """
    visit_probability = tf.reduce_mean(
        p, [0], keep_dims=True, name='visit_prob')
    t_nb = tf.shape(p)[1]
    visit_loss = tf.losses.softmax_cross_entropy(
        tf.fill([1, t_nb], 1.0 / tf.cast(t_nb, tf.float32)),
        tf.log(1e-8 + visit_probability),
        weights=weight,
        scope='loss_visit')

    tf.summary.scalar('Loss_Visit', visit_loss)

  def add_logit_loss(self, logits, labels, weight=1.0, smoothing=0.0):
    """Add supervised classification loss to the model."""

    logit_loss = tf.losses.softmax_cross_entropy(
        tf.one_hot(labels, logits.get_shape()[-1]),
        logits,
        scope='loss_logit',
        weights=weight,
        label_smoothing=smoothing)

    tf.summary.scalar('Loss_Logit', logit_loss)

  def create_walk_statistics(self, p_aba, equality_matrix):
    """Adds "walker" loss statistics to the graph.

    Args:
      p_aba: [N, N] matrix, where element [i, j] corresponds to the
          probalility of the round-trip between supervised samples i and j.
          Sum of each row of 'p_aba' must be equal to one.
      equality_matrix: [N, N] boolean matrix, [i,j] is True, when samples
          i and j belong to the same class.
    """
    # Using the square root of the correct round trip probalilty as an estimate
    # of the current classifier accuracy.
    per_row_accuracy = 1.0 - tf.reduce_sum((equality_matrix * p_aba), 1)**0.5
    estimate_error = tf.reduce_mean(
        1.0 - per_row_accuracy, name=p_aba.name[:-2] + '_esterr')
    self.add_average(estimate_error)
    self.add_average(p_aba)

    tf.summary.scalar('Stats_EstError', estimate_error)

  def add_average(self, variable):
    """Add moving average variable to the model."""
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, self.ema.apply([variable]))
    average_variable = tf.identity(
        self.ema.average(variable), name=variable.name[:-2] + '_avg')
    return average_variable

  def create_train_op(self, learning_rate):
    """Create and return training operation."""

    slim.model_analyzer.analyze_vars(
        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), print_info=True)

    self.train_loss = tf.losses.get_total_loss()
    self.train_loss_average = self.add_average(self.train_loss)

    tf.summary.scalar('Learning_Rate', learning_rate)
    tf.summary.scalar('Loss_Total_Avg', self.train_loss_average)
    tf.summary.scalar('Loss_Total', self.train_loss)

    trainer = tf.train.AdamOptimizer(learning_rate)

    self.train_op = slim.learning.create_train_op(self.train_loss, trainer)
    return self.train_op, self.train_loss

  def calc_embedding(self, images, endpoint):
    """Evaluate 'endpoint' tensor for all 'images' using batches."""
    batch_size = self.test_batch_size
    emb = []
    for i in xrange(0, len(images), batch_size):
      emb.append(endpoint.eval({self.test_in: images[i:i + batch_size]}))
    return np.concatenate(emb)

  def classify(self, images):
    """Compute logit scores for provided images."""
    return self.calc_embedding(images, self.test_logit)
 
  def classify_loss(self, images, labels):
    """Compute logit scores for provided images."""
    batch_size = self.test_batch_size
    loss_1 = []
    for i in xrange(0, len(images), batch_size):
      loss_1.append(self.test_loss.eval({self.test_in: images[1:1 + batch_size], self.test_label: labels[1:1 + batch_size].reshape(-1,)}))
      loss_test = np.mean(loss_1,axis=0)    
    return loss_test

  def embeddings(self, images):
    batch_size = 1
    embedding = self.test_emb.eval({self.test_in:images})
    return embedding
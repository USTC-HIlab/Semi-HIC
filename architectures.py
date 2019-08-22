from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim


def dataset3_model(inputs,
                  is_training=True,
                  emb_size=128,
                  l2_weight=1e-3,#1e-3
                  batch_norm_decay=None,
                  img_shape=[64,64,3],
                  new_shape=None,
                  augmentation_function=None,
                  image_summary=False):  # pylint: disable=unused-argument

    """Construct the image-to-embedding vector model."""

    inputs = tf.cast(inputs, tf.float32) / 255.0
    if new_shape is not None:
        shape = new_shape
        inputs = tf.image.resize_images(
            inputs,
            tf.constant(new_shape[:2]),
            method=tf.image.ResizeMethod.BILINEAR)
    else:
        shape = img_shape
    net = inputs
    mean = tf.reduce_mean(net, [1, 2], True)
    std = tf.reduce_mean(tf.square(net - mean), [1, 2], True)
    net = (net - mean) / (std + 1e-5)

    with slim.arg_scope(
            [slim.conv2d, slim.fully_connected],
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            normalizer_params={'is_training': is_training},
            weights_regularizer=slim.l2_regularizer(l2_weight)):
        net = slim.conv2d(net, 16, [3, 3], scope='conv1_1')
        net = slim.conv2d(net, 16, [3, 3], scope='conv1_2')
        net = slim.conv2d(net, 16, [2, 2], stride=2, padding='VALID', scope='conv1_3')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        

        net = slim.conv2d(net, 32, [3, 3], scope='conv2_1')
        net = slim.conv2d(net, 32, [3, 3], scope='conv2_2')
        net = slim.conv2d(net, 32, [2, 2], stride=2, padding='VALID', scope='conv2_3')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')

        net = slim.conv2d(net, 64, [3, 3], scope='conv3_1')
        net = slim.conv2d(net, 64, [3, 3], scope='conv3_2')
        net = slim.conv2d(net, 64, [2, 2], stride=2, padding='VALID', scope='conv3_3')
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        
        net = slim.conv2d(net, 128, [3, 3], scope='conv4_1')
        net = slim.conv2d(net, 128, [3, 3], scope='conv4_2')
        net = slim.conv2d(net, 128, [3, 3], scope='conv4_3')
#         net = slim.max_pool2d(net, [3, 3], scope='pool4')
        
        net = slim.conv2d(net, 256, [3, 3], scope='conv5_1')
        net = slim.conv2d(net, 256, [3, 3], scope='conv5_2')
        net = slim.conv2d(net, 256, [1, 1], scope='conv5_3')
#         net = slim.max_pool2d(net, [1, 1], scope='pool5')

        net = slim.flatten(net, scope='flatten')
#         emb = slim.fully_connected(net, emb_size, scope='fc1')
        emb = slim.fully_connected(net, emb_size, normalizer_fn=None, scope='fc1')
    return emb


def dataset_model(inputs,
                  is_training=True,
                  emb_size=128,
                  l2_weight=1e-3,#1e-3,
                  batch_norm_decay=None,
                  img_shape=None,
                  new_shape=None,
                  augmentation_function=None,
                  image_summary=False):  # pylint: disable=unused-argument

    """Construct the image-to-embedding vector model."""

    inputs = tf.cast(inputs, tf.float32) / 255.0
    if new_shape is not None:
        shape = new_shape
        inputs = tf.image.resize_images(
            inputs,
            tf.constant(new_shape[:2]),
            method=tf.image.ResizeMethod.BILINEAR)
    else:
        shape = img_shape
    net = inputs
    with slim.arg_scope(
            [slim.conv2d, slim.fully_connected],
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            normalizer_params={'is_training': is_training},
            weights_regularizer=slim.l2_regularizer(l2_weight)):
        net = slim.conv2d(net, 16, [3, 3], scope='conv1_1')
        net = slim.conv2d(net, 16, [3, 3], scope='conv1_2')
        net = slim.conv2d(net, 16, [2, 2], stride=2, padding='VALID', scope='conv1_3')

        net = slim.conv2d(net, 32, [3, 3], scope='conv2_1')
        net = slim.conv2d(net, 32, [3, 3], scope='conv2_2')
        net = slim.conv2d(net, 32, [2, 2], stride=2, padding='VALID', scope='conv2_3')

        net = slim.conv2d(net, 64, [3, 3], scope='conv3_1')
        net = slim.conv2d(net, 64, [3, 3], scope='conv3_2')
        net = slim.conv2d(net, 64, [2, 2], stride=2, padding='VALID', scope='conv3_3')
        
        net = slim.conv2d(net, 128, [3, 3], scope='conv4_1')
        net = slim.conv2d(net, 128, [3, 3], scope='conv4_2')
        net = slim.conv2d(net, 128, [3, 3], scope='conv4_3')
        
        net = slim.conv2d(net, 256, [3, 3], scope='conv5_1')
        net = slim.conv2d(net, 256, [3, 3], scope='conv5_2')
        net = slim.conv2d(net, 256, [1, 1], scope='conv5_3')        

        net = slim.flatten(net, scope='flatten')
        emb = slim.fully_connected(net, emb_size, scope='fc1')
    return emb

def dataset2_model(inputs,
                  is_training=True,
                  emb_size=128,
                  l2_weight=1e-3,
                  batch_norm_decay=None,
                  img_shape=None,
                  new_shape=None,
                  augmentation_function=None,
                  image_summary=False):  # pylint: disable=unused-argument

    """Construct the image-to-embedding vector model."""

    inputs = tf.cast(inputs, tf.float32) / 255.0
    if new_shape is not None:
        shape = new_shape
        inputs = tf.image.resize_images(
            inputs,
            tf.constant(new_shape[:2]),
            method=tf.image.ResizeMethod.BILINEAR)
    else:
        shape = img_shape
    net = inputs
    with slim.arg_scope(
            [slim.conv2d, slim.fully_connected],
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            normalizer_params={'is_training': is_training},
            weights_regularizer=slim.l2_regularizer(l2_weight)):
        net = slim.conv2d(net, 16, [3, 3], scope='conv1_1')
        net = slim.conv2d(net, 16, [3, 3], scope='conv1_2')
        net = slim.conv2d(net, 16, [2, 2], stride=2, padding='VALID', scope='conv1_3')

        net = slim.conv2d(net, 32, [3, 3], scope='conv2_1')
        net = slim.conv2d(net, 32, [3, 3], scope='conv2_2')
        net = slim.conv2d(net, 32, [2, 2], stride=2, padding='VALID', scope='conv2_3')

        net = slim.conv2d(net, 64, [3, 3], scope='conv3_1')
        net = slim.conv2d(net, 64, [3, 3], scope='conv3_2')
        net = slim.conv2d(net, 64, [2, 2], stride=2, padding='VALID', scope='conv3_3')
        
        net = slim.conv2d(net, 128, [3, 3], scope='conv4_1')
        net = slim.conv2d(net, 128, [3, 3], scope='conv4_2')
        net = slim.conv2d(net, 128, [2, 2], stride=2, padding='VALID', scope='conv4_3')
                
        net = slim.conv2d(net, 256, [3, 3], scope='conv5_1')
        net = slim.conv2d(net, 256, [3, 3], scope='conv5_2')
        net = slim.conv2d(net, 256, [3, 3], scope='conv5_3')
        
        net = slim.conv2d(net, 512, [3, 3], scope='conv6_1')
        net = slim.conv2d(net, 512, [3, 3], scope='conv6_2')
        net = slim.conv2d(net, 512, [1, 1], scope='conv6_3')

        net = slim.flatten(net, scope='flatten')
        emb = slim.fully_connected(net, emb_size, scope='fc1')
    return emb


def minist_model(inputs,
                 is_training=True,
                 emb_size=128,
                 l2_weight=1e-3,
                 batch_norm_decay=None,
                 img_shape=None,
                 new_shape=None,
                 augmentation_function=None,
                 image_summary=False):  # pylint: disable=unused-argument

    """Construct the image-to-embedding vector model."""

    inputs = tf.cast(inputs, tf.float32) / 255.0
    if new_shape is not None:
        shape = new_shape
        inputs = tf.image.resize_images(
            inputs,
            tf.constant(new_shape[:2]),
            method=tf.image.ResizeMethod.BILINEAR)
    else:
        shape = img_shape
    net = inputs
    with slim.arg_scope(
            [slim.conv2d, slim.fully_connected],
            activation_fn=tf.nn.elu,
            weights_regularizer=slim.l2_regularizer(l2_weight)):
        net = slim.conv2d(net, 32, [3, 3], scope='conv1_1')
        net = slim.conv2d(net, 32, [3, 3], scope='conv1_2')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')  # 14

        net = slim.conv2d(net, 64, [3, 3], scope='conv2_1')
        net = slim.conv2d(net, 64, [3, 3], scope='conv2_2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')  # 7

        net = slim.conv2d(net, 128, [3, 3], scope='conv3_1')
        net = slim.conv2d(net, 128, [3, 3], scope='conv3_2')
        net = slim.max_pool2d(net, [2, 2], scope='pool3')  # 3

        net = slim.flatten(net, scope='flatten')
        emb = slim.fully_connected(net, emb_size, scope='fc1')
    return emb


def inception_model(inputs,
                    emb_size=128,
                    is_training=True,
                    end_point='Mixed_7c',
                    augmentation_function=None,
                    img_shape=None,
                    new_shape=None,
                    batch_norm_decay=None,
                    dropout_keep_prob=0.8,
                    min_depth=16,
                    depth_multiplier=1.0,
                    spatial_squeeze=True,
                    reuse=None,
                    scope='InceptionV3',
                    num_classes=10,
                    **kwargs):
    from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3_base
    from tensorflow.python.ops import variable_scope
    from tensorflow.contrib.framework.python.ops import arg_scope
    from tensorflow.contrib.layers.python.layers import layers as layers_lib

    inputs = tf.cast(inputs, tf.float32) / 255.0
    if new_shape is not None:
        shape = new_shape
        inputs = tf.image.resize_images(
            inputs,
            tf.constant(new_shape[:2]),
            method=tf.image.ResizeMethod.BILINEAR)
    else:
        shape = img_shape

    net = inputs
    mean = tf.reduce_mean(net, [1, 2], True)
    std = tf.reduce_mean(tf.square(net - mean), [1, 2], True)
    net = (net - mean) / (std + 1e-5)

    inputs = net

    if depth_multiplier <= 0:
        raise ValueError('depth_multiplier is not greater than zero.')

    with variable_scope.variable_scope(
            scope, 'InceptionV3', [inputs, num_classes], reuse=reuse) as scope:
        with arg_scope(
                [layers_lib.batch_norm, layers_lib.dropout], is_training=is_training):
            _, end_points = inception_v3_base(
                inputs,
                scope=scope,
                min_depth=min_depth,
                depth_multiplier=depth_multiplier,
                final_endpoint=end_point)

    net = end_points[end_point]
    net = slim.flatten(net, scope='flatten')
    with slim.arg_scope([slim.fully_connected], normalizer_fn=None):
        emb = slim.fully_connected(net, emb_size, scope='fc')
    return emb


def inception_model_small(inputs,
                          emb_size=128,
                          is_training=True,
                          **kwargs):
    return inception_model(inputs=inputs, emb_size=emb_size, is_training=is_training,
                           end_point='Mixed_5d', **kwargs)
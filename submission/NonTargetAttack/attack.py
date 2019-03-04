"""
Implementation of example defense.
This defense loads inception v1 checkpoint and classifies all images using loaded checkpoint.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import tensorflow as tf

import sys
sys.path.append("../../")
print(sys.path)

from IJCAI19.module.utils import *
from IJCAI19.module.utils_tf import * 
from IJCAI19.module.EmbeddedAttackModel import TargetModel, EmbeddedAttackModel
from IJCAI19.module.gs_mim import GradSmoothMomentumIterativeMethod


tf.flags.DEFINE_string(
    'weight_path', 'IJCAI19/weight', 'Path to checkpoint for inception network.')
tf.flags.DEFINE_string(
    'input_dir', '', 'Input directory with images.')
tf.flags.DEFINE_string(
    'output_dir', '', 'Output directory with images.')
tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')
tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')
tf.flags.DEFINE_integer(
    'batch_size', 16, 'How many images process at one time.')
tf.flags.DEFINE_integer(
    'num_classes', 110, 'Number of Classes')
FLAGS = tf.flags.FLAGS

def attack():
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]

    img_loader = ImageLoader(FLAGS.input_dir, batch_shape, label_size=None, format='png', labels=None)
    img_saver = ImageSaver(FLAGS.output_dir, save_format='png', save_prefix='', scale=False)

    name = 'inception_v1'
    T1 = TargetModel(batch_shape, FLAGS.num_classes, name=name)
    name = 'resnetv1_50'
    T2 = TargetModel(batch_shape, FLAGS.num_classes, name=name)
    name = 'vgg_16'
    T3 = TargetModel(batch_shape, FLAGS.num_classes, name=name)

    A = EmbeddedAttackModel(batch_shape, FLAGS.num_classes)
    A.add_model(T1)

    M = GradSmoothMomentumIterativeMethod
    attack_params = {"ep_ratio": 0.1, "nb_iter": 10}

    config = gpu_session_config()
    with tf.Session(config=config) as sess:
        A.attack_generate(sess, M, attack_params)
        for filenames, X, Y in img_loader:
            Xadv = A.attack_batch(X, Y)
            for i in range(Xadv.shape[0]):
                sample_size+=1
                fname = filenames[i][:15]
                AdvSaver.save_array(fname, Xadv[i])
            if sample_size > max_sample_size:
                duration = time.time() - check_timestamp
                check_timestamp = time.time()
                print("%s total %d, duration %.2f mins" %(prefix, sample_size,duration/60))
                break
    tf.reset_default_graph()

def main(_):

    tf.logging.set_verbosity(tf.logging.INFO)
    print("done")




if __name__ == '__main__':
    tf.app.run()

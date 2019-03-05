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
from IJCAI19.model.EmbeddedAttackModel import TargetModel, EmbeddedAttackModel
from cleverhans.attacks import FastGradientMethod
from IJCAI19.module.gs_mim import GradSmoothMomentumIterativeMethod
from IJCAI19.model.OfficialModel import OfficialModel

tf.flags.DEFINE_string(
    'weight_path', 'IJCAI19/weight/', 'Path to checkpoint for inception network.')
tf.flags.DEFINE_string(
    'input_dir', '', 'Input directory with images.')
tf.flags.DEFINE_string(
    'output_dir', '', 'Output directory with images.')
tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')
tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')
tf.flags.DEFINE_integer(
    'batch_size', 8, 'How many images process at one time.')
tf.flags.DEFINE_integer(
    'num_classes', 110, 'Number of Classes')
FLAGS = tf.flags.FLAGS

tf.app.flags.DEFINE_string('f', '', 'kernel')

def attack(M, attack_params, targetlabel):
    OfficialModel.WEIGHT_DIR = FLAGS.weight_path
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]

    # img_loader = ImageLoader(FLAGS.input_dir, batch_shape, label_size=None, format='png', labels=None)
    img_loader = ImageLoader(FLAGS.input_dir, batch_shape, targetlabel=targetlabel, label_size=FLAGS.num_classes, format='png', label_file='dev.csv')
    img_saver = ImageSaver(FLAGS.output_dir, save_format='png', save_prefix='', scale=False)

    name = 'inception_v1'
    T1 = TargetModel(batch_shape, FLAGS.num_classes, name=name)
    name = 'resnetv1_50'
    T2 = TargetModel(batch_shape, FLAGS.num_classes, name=name)
    name = 'vgg_16'
    T3 = TargetModel(batch_shape, FLAGS.num_classes, name=name)

    A = EmbeddedAttackModel(batch_shape, FLAGS.num_classes)
    A.add_model(T1)
    A.add_model(T2)
    # A.add_model(T3)

    config = gpu_session_config()
    with tf.Session(config=config) as sess:
        A.attack_generate(sess, M, attack_params)
        for filenames, X, Y in img_loader:
            Xadv = A.attack_batch(X, Y)
            for i in range(Xadv.shape[0]):
                img_saver.save_array(filenames[i], Xadv[i])
    tf.reset_default_graph()


def main(_):
    USE_TRUE_TARGET = True
    tf.logging.set_verbosity(tf.logging.WARNING)
    # M = GradSmoothMomentumIterativeMethod
    M = FastGradientMethod
    #non targeted with guessed label
    attack_params = {"ep_ratio": 0.1, "nb_iter": 10, "y":USE_TRUE_TARGET}
    attack(M, attack_params, targetlabel=False)

    print("done")


if __name__ == '__main__':
    tf.app.run()

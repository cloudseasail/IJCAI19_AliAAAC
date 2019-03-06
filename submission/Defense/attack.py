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

# import sys
# sys.path.append("../../")
# print(sys.path)

from IJCAI19.module.utils import *
from IJCAI19.module.utils_tf import * 
from IJCAI19.model.EmbeddedDefenseModel import *
from IJCAI19.model.ModelFactory import ModelFactory

tf.flags.DEFINE_string(
    'weight_path', 'IJCAI19/weight/', 'Path to checkpoint for inception network.')
tf.flags.DEFINE_string(
    'input_dir', '', 'Input directory with images.')
tf.flags.DEFINE_string(
    'output_file', '', 'Output file to save labels.')
tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')
tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')
tf.flags.DEFINE_integer(
    'batch_size', 8, 'How many images process at one time.')
tf.flags.DEFINE_integer(
    'num_classes', 110, 'How many classes of the data set')
FLAGS = tf.flags.FLAGS

tf.app.flags.DEFINE_string('f', '', 'kernel')

def defense(D):
    ModelFactory.WEIGHT_DIR = FLAGS.weight_path
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]

    # img_loader = ImageLoader(FLAGS.input_dir, batch_shape, label_size=None, format='png', labels=None)
    img_loader = ImageLoader(FLAGS.input_dir, batch_shape, targetlabel=False, label_size=FLAGS.num_classes, format='png', label_file=None)

    config = gpu_session_config()
    with tf.Session(config=config) as sess:
        D.predict_generate(sess)
        with open(FLAGS.output_file, 'w') as out_file:
            for filenames, X, _ in img_loader:
                ypred = D.predict_batch(X, None)
                for filename, label in zip(filenames, ypred.argmax(1)):
                    out_file.write('{0},{1}\n'.format(filename, label))

    tf.reset_default_graph()


def main(_):

    tf.logging.set_verbosity(tf.logging.WARN)

    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    name = "inception_v1"
    D = MSBModel(msb=8, batch_shape=batch_shape, output_size=FLAGS.num_classes, name=name, use_prob=True)
    defense(D)

    print("done")


if __name__ == '__main__':
    tf.app.run()

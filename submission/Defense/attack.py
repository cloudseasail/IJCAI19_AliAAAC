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

    img_loader = ImageLoader(FLAGS.input_dir, batch_shape, targetlabel=False, label_size=FLAGS.num_classes, format='png', label_file=None)


    with open(FLAGS.output_file, 'w') as out_file:
        for filenames, X, _ in img_loader:
            ypred = D.predict_batch(X, None)
            for filename, label in zip(filenames, ypred):
                out_file.write('{0},{1}\n'.format(filename, label))

    D.clear_session()


def main(_):

    tf.logging.set_verbosity(tf.logging.WARN)

    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    name = "keras_xception_19"
    D = ModelFactory.create(name=name, nb_classes=FLAGS.num_classes)
    defense(D)

    D.clear_session()

    print("done")


if __name__ == '__main__':
    tf.app.run()

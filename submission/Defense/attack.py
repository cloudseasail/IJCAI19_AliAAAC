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
from IJCAI19.model.RandomDefense import RandomDefense

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
ModelFactory.WEIGHT_DIR = FLAGS.weight_path

def defense(D, repeat=1):
    ModelFactory.WEIGHT_DIR = FLAGS.weight_path
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    img_loader = ImageLoader(FLAGS.input_dir, batch_shape, targetlabel=False, label_size=FLAGS.num_classes, format='png', label_file=None)

    D.predict_create_graph(batch_shape)
    with open(FLAGS.output_file, 'w') as out_file:
        for filenames, X, _ in img_loader:
            ypred = D.predict_batch(X, repeat)
            ypred = ypred.argmax(1)
            for filename, label in zip(filenames, ypred):
                out_file.write('{0},{1}\n'.format(filename, label))
                # print(filename, label)


def main(_):

    tf.logging.set_verbosity(tf.logging.WARN)

    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    name = "inception_v1"
    T1 = RandomDefense(FLAGS.num_classes, name=name)
    T1.random(
            msb_max=16, msb_rate=1, 
            rotation_range=20,
            width_shift_range=0.05,
            height_shift_range=0.05,
    #         shear_range=0.05
    )
    name = "resnetv1_50"
    T2 = RandomDefense(FLAGS.num_classes, name=name)
    T2.random(
            msb_max=16, msb_rate=1, 
            rotation_range=20,
            width_shift_range=0.05,
            height_shift_range=0.05,
    #         shear_range=0.05
    )
    name = "vgg_16"
    T3 = RandomDefense(FLAGS.num_classes, name=name)
    T3.random(
            msb_max=16, msb_rate=1, 
            rotation_range=20,
            width_shift_range=0.05,
            height_shift_range=0.05,
    #         shear_range=0.05
    )

    name = "keras_xception_19"
    T11 = RandomDefense(FLAGS.num_classes, name=name)
    T11.random(
            msb_max=8, msb_rate=1, 
            rotation_range=10,
    #         width_shift_range=0.05,
    #         height_shift_range=0.05
    )

    name = "keras_nasnet_large"
    T12 = RandomDefense(FLAGS.num_classes, name=name)
    # T12.random(
    #         msb_max=8, msb_rate=1, 
    #         rotation_range=10
    # )


    D = EmbeddedDefenseModel("")
    # D.add_model(T1, weight=1)
    # D.add_model(T2, weight=1)
    # D.add_model(T3, weight=1)
    # D.add_model(T11, weight=1)
    D.add_model(T12, weight=1)

    defense(D, repeat=1)

    print("done")


if __name__ == '__main__':
    tf.app.run()

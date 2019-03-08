import os
import numpy as np
import tensorflow as tf
import csv

# import sys
# sys.path.append("../")
# print(sys.path)

from IJCAI19.module.utils import *
from IJCAI19.module.utils_tf import * 
from IJCAI19.model.EmbeddedAttackModel import AttackModel, EmbeddedAttackModel
from cleverhans.attacks import FastGradientMethod
from IJCAI19.module.gs_mim import GradSmoothMomentumIterativeMethod
from IJCAI19.model.ModelFactory import ModelFactory

FLAGS = tf.flags.FLAGS

def InitGlobal(): 
    tf.flags.DEFINE_string(
    'weight_path', 'IJCAI19/weight/', 'Path to checkpoint for inception network.')
    tf.flags.DEFINE_string(
        'dev_dir', '', 'dev directory with images.')
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

    FLAGS.dev_dir = "../../official_data/dev_data/"
    FLAGS.input_dir = "../../official_data/dev_data/"
    FLAGS.output_file = "../../test_data/DefenseResult.csv"
    FLAGS.weight_path = "../IJCAI19/weight/"
    FLAGS.batch_size = 4

    ModelFactory.WEIGHT_DIR = FLAGS.weight_path

    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]

def Defense(D, path, repeat=1):
    ModelFactory.WEIGHT_DIR = FLAGS.weight_path
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    p=Profile('Defense ')
    img_loader = ImageLoader(path, batch_shape, targetlabel=False, label_size=FLAGS.num_classes, format='png', label_file='dev.csv')
    ypred, yprob = D.predict_generator(img_loader, batch_shape, repeat=repeat)
    p.stop()
    return ypred, yprob

def DefenseWrite(D, repeat=1):
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

def Validate():
    all_shape = [110, FLAGS.image_height, FLAGS.image_width, 3]
    results, length = LoadResult()

    img_loader = ImageLoader(FLAGS.dev_dir, all_shape, targetlabel=False, label_size=FLAGS.num_classes, format='png', label_file='dev.csv')
    filenames, X, Y = next(img_loader)
    img_loader = ImageLoader(FLAGS.input_dir, all_shape, targetlabel=False, label_size=FLAGS.num_classes, format='png', label_file='dev.csv')
    filenames, Xadv, Y = next(img_loader)
    Y = Y.argmax(1)

    if length != len(filenames):
        print("length not match", length, len(filenames))

    Score = 0
    correct = 0
    for i in range(length):
        name = filenames[i]
        if (Y[i] == results[name]):
            Score += calc_l2(X[i], Xadv[i])
            correct += 1
    Score = Score/length
    print("correct {0}/{1},  Score {2}".format(correct, length, Score))
    return Score


def LoadResult():
    result = {}
    length = 0
    with open(FLAGS.output_file) as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            result[row[0]] = int(row[1])
            length += 1
    return result, length
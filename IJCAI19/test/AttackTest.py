import os
import numpy as np
import tensorflow as tf

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
batch_shape = None

def InitGlobal(): 
    global batch_shape
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

    tf.app.flags.DEFINE_string('f', '', 'kernel')

    FLAGS.input_dir = "../../official_data/dev_data/"
    FLAGS.output_dir = "../../test_data/NonTargetAttackResult/"
    FLAGS.weight_path = "../IJCAI19/weight/"
    FLAGS.batch_size = 4

    ModelFactory.WEIGHT_DIR = FLAGS.weight_path
    open(FLAGS.output_dir+'dev.csv', "wb").write(open(FLAGS.input_dir+'dev.csv', "rb").read())

    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]

def Attack(A, M, attack_params, targetlabel):
    ModelFactory.WEIGHT_DIR = FLAGS.weight_path
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    p=Profile('Attack ')
    # img_loader = ImageLoader(FLAGS.input_dir, batch_shape, label_size=None, format='png', labels=None)
    img_loader = ImageLoader(FLAGS.input_dir, batch_shape, targetlabel=targetlabel, label_size=FLAGS.num_classes, format='png', label_file='dev.csv')
    img_saver = ImageSaver(FLAGS.output_dir, save_format='png', save_prefix='', scale=False)

    config = gpu_session_config()
    sess = tf.Session(config=config)
    A.attack_generate(sess, M, attack_params)
    for filenames, X, Y in img_loader:
        Xadv = A.attack_batch(X, Y)
        for i in range(Xadv.shape[0]):
            img_saver.save_array(filenames[i], Xadv[i])
    sess.close()
    tf.reset_default_graph()
    p.stop()

def Predict(T, dir, targetlabel):
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    img_loader = ImageLoader(dir, batch_shape, targetlabel=targetlabel, label_size=FLAGS.num_classes, format='png', label_file='dev.csv')
    Yp, topK, acc = T.evaluate_generator(img_loader, batch_shape, use_prob=False)
    return Yp, topK, acc

def Score(Yp, targetlabel):
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    all_shape = (110, batch_shape[1], batch_shape[2], batch_shape[3])
    img_loader = ImageLoader(FLAGS.input_dir, all_shape, targetlabel=False, label_size=FLAGS.num_classes, format='png', label_file='dev.csv')
    _, X, Y = next(img_loader)
    img_loader = ImageLoader(FLAGS.output_dir, all_shape, targetlabel=targetlabel, label_size=FLAGS.num_classes, format='png', label_file='dev.csv')
    _, Xadv, Yadv = next(img_loader)

    print(X.shape, Xadv.shape, Y.shape, Yadv.shape)
    score, succ =  calc_score(X, Xadv, Yadv.argmax(1), Yp, target=targetlabel)
    print("Mean L2 %.4f,  Score %.4f, Attack Success Rate %.4f" % (calc_l2(X, Xadv), score, succ ))
    return X, Y, Xadv, Yadv


def calc_score_slow(x, xadv, y, yadv):
    score = 0
    for i in range(x.shape[0]):
        if y[i] == yadv[i]:
            score += 128
        else:
            score += calc_l2(x,xadv)
    return score/x.shape[0]
def calc_score(x, xadv, y, yadv, target=False):
    if target:
        succ = (y == yadv)
    else:
        succ = (y != yadv)
    succ_num = x[succ].shape[0]
    succ_mean = calc_l2(x[succ],xadv[succ])
    succ_score = succ_mean*succ_num
    fail_score = 128* (x.shape[0] - succ_num)
    # print(succ_num, succ_mean, fail_score)
    score = (succ_score+fail_score)/x.shape[0]
    succ = succ_num/x.shape[0]
    return score, succ
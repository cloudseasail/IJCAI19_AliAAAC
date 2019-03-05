import tensorflow as tf
slim = tf.contrib.slim
import numpy as np
from tensorflow.contrib.slim.python.slim.nets import inception_v1, resnet_v1
from . import slimvgg as vgg
from tensorflow.contrib import slim
from tensorflow.contrib.layers.python.layers import layers as layers_lib

# WEIGHT_DIR = '../official_data/'

def Inception_preprocess(imgs, undo=False):
    if (undo):
        out_imgs = (((imgs + 1.0) * 0.5) * 255.0)
    else:
        out_imgs = (imgs / 255.0) * 2.0 - 1.0
    return out_imgs


def Vgg_preprocess_tf(imgs, undo=False):
    _R_MEAN = 123.68
    _G_MEAN = 116.78
    _B_MEAN = 103.94
    means= [_R_MEAN, _G_MEAN, _B_MEAN]
    channels = tf.split(axis=3, num_or_size_splits=3, value=imgs)
    for i in range(3):
        if (undo):
            channels[i] += means[i]
        else:
            channels[i] -= means[i]
    out_imgs = tf.concat(axis=3, values=channels)
    return out_imgs

class OfficialModel():
    WEIGHT_DIR = '../weight/'
    model = {
        'inception_v1': {
            'var_scope': 'InceptionV1',
            'arg_scope': inception_v1.inception_v1_arg_scope,
            'graph': inception_v1.inception_v1,
            'checkpoint_dir':  'inception_v1/inception_v1.ckpt',
            'preprocess': Inception_preprocess,
            'min_max': (-1.0, 1.0),
            'default_input_size': 224,
        },
        'resnetv1_50': {
            'var_scope': 'resnet_v1_50',
            'arg_scope': resnet_v1.resnet_arg_scope,
            'graph': resnet_v1.resnet_v1_50,
            'checkpoint_dir':  'resnet_v1_50/model.ckpt-49800',
            'preprocess': Vgg_preprocess_tf,
            'min_max': (0-115, 255-115),
            'default_input_size': 224,
        },
        'vgg_16': {
            'var_scope': 'vgg_16',
            'arg_scope': vgg.vgg_arg_scope,
            'graph': vgg.vgg_16,
            'checkpoint_dir':  'vgg_16/vgg_16.ckpt',
            'preprocess': Vgg_preprocess_tf,
            'min_max': (0-115, 255-115),
            'default_input_size': 224,
        }
    }
    def __init__(self, name):
        self.name = name
        self._model = OfficialModel.model[name]
        # self.weight_loaded = False

    def get_endpoints(self, x, nb_classes):
        with slim.arg_scope(self._model['arg_scope']()):
            # print("get_endpoints", self.name)
            logits, end_points = self._model['graph'](x, num_classes=nb_classes,is_training=False,reuse=tf.AUTO_REUSE)
            if 'Logits' not in end_points:
                end_points['Logits'] = logits
            if (len(end_points['Logits'].shape) == 4):
                end_points['Logits'] = tf.squeeze(end_points['Logits'] , [1,2])
            if 'Predictions' not in end_points:
                end_points['Predictions'] = layers_lib.softmax(end_points['Logits'], scope='predictions')
        return end_points
    def _get_weight(self):
        return OfficialModel.WEIGHT_DIR + self._model['checkpoint_dir']
    def load_weight(self, checkpoint_path=''):
        # if self.weight_loaded == False:
        if True:
            saver = tf.train.Saver(slim.get_model_variables(scope=self._model['var_scope']))
            if checkpoint_path == '':
                checkpoint_path = self._get_weight()
            saver.restore(tf.get_default_session(), checkpoint_path)
            # saver.restore(self.attack_sess, checkpoint_path)
            # self.weight_loaded = True
    def _input_resize(self, imgs):
        default_input_size = self._model['default_input_size']
        imgs = tf.image.resize_images(imgs, [default_input_size,default_input_size])
        return imgs
    def output_resize(self, imgs, size):
        imgs = tf.image.resize_images(imgs, [size,size])
        return imgs
    def preprocess(self, imgs):
        return self._model['preprocess'](imgs)
    def attack_preprocess(self, imgs):
        imgs = self._input_resize(imgs)
        return self.preprocess(imgs)
    def predict_preprocess(self, imgs):
        imgs = self._input_resize(imgs)
        return self.preprocess(imgs)




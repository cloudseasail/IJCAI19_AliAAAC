import tensorflow as tf
slim = tf.contrib.slim
import numpy as np
from tensorflow.contrib.slim.python.slim.nets import inception_v1, resnet_v1
from . import slimvgg as vgg
from tensorflow.contrib import slim
from tensorflow.contrib.layers.python.layers import layers as layers_lib

def Inception_preprocess(imgs, undo=False):
    if (undo):
        out_imgs = (((imgs + 1.0) * 0.5) * 255.0)
    else:
        out_imgs = (imgs / 255.0) * 2.0 - 1.0
    return out_imgs

def Vgg_preprocess_np(imgs, undo=False):
    _R_MEAN = 123.68
    _G_MEAN = 116.78
    _B_MEAN = 103.94
    means= [_R_MEAN, _G_MEAN, _B_MEAN]
    out_imgs = imgs.copy()
    channels = np.split(out_imgs, 3, axis=3)
    for i in range(3):
        if (undo):
            channels[i] += means[i]
        else:
            channels[i] -= means[i]
    out_imgs = np.concatenate(channels, axis=3)
    # if (undo):
    #     out_imgs = out_imgs.astype(np.uint8)
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
    root = '../official_data/'
    model = {
        'inception_v1': {
            'var_scope': 'InceptionV1',
            'arg_scope': inception_v1.inception_v1_arg_scope,
            'graph': inception_v1.inception_v1,
            'checkpoint_dir': root + 'model/inception_v1/inception_v1.ckpt',
            'preprocess': Inception_preprocess,
            'min_max': (-1.0, 1.0)
        },
        'resnetv1_50': {
            'var_scope': 'resnet_v1_50',
            'arg_scope': resnet_v1.resnet_arg_scope,
            'graph': resnet_v1.resnet_v1_50,
            'checkpoint_dir': root + 'model/resnet_v1_50/model.ckpt-49800',
            'preprocess': Vgg_preprocess_tf,
            'min_max': (0-115, 255-115)
        },
        'vgg_16': {
            'var_scope': 'vgg_16',
            'arg_scope': vgg.vgg_arg_scope,
            'graph': vgg.vgg_16,
            'checkpoint_dir': root + 'model/vgg_16/vgg_16.ckpt',
            'preprocess': Vgg_preprocess_tf,
            'min_max': (0-115, 255-115)
        }
    }
    def __init__(self, name):
        self.name = name
        self._model = OfficialModel.model[name]
        self.weight_loaded = False
        self.built = False
    def loaded(self):
        graph = tf.get_default_graph()
        loaded = False
        if hasattr(graph, "_official_model_built_"):
            if self.name in graph._official_model_built_:
                loaded = True
        else:
            graph._official_model_built_ = []

        if (loaded == False):
            graph._official_model_built_.append(self.name)
        return loaded
    def get_endpoints(self, x, nb_classes):
        # reuse = self.loaded()
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
    def get_weight(self):
        return self._model['checkpoint_dir']
    def load_weight(self, checkpoint_path=''):
        # if self.weight_loaded == False:
        if True:
            saver = tf.train.Saver(slim.get_model_variables(scope=self._model['var_scope']))
            if checkpoint_path == '':
                checkpoint_path = self.get_weight()
            saver.restore(tf.get_default_session(), checkpoint_path)
            # saver.restore(self.attack_sess, checkpoint_path)
            # self.weight_loaded = True
    def preprocess(self, imgs):
        return self._model['preprocess'](imgs)
    def attack_preprocess(self, imgs):
        return self.preprocess(imgs)
        # if (self.name == 'resnetv1_50') or self.name == 'vgg_16':
        #     return imgs-115
        # else:
        #     return self._model['preprocess'](imgs)




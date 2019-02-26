import tensorflow as tf
import numpy as np
from tensorflow.contrib.slim.python.slim.nets import inception_v1, resnet_v1, vgg
# from keras.applications import inception_v3, resnet50
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
    channels = np.split(imgs, 3, axis=3)
    for i in range(3):
        if (undo):
            channels[i] += means[i]
        else:
            channels[i] -= means[i]
    out_imgs = np.concatenate(channels, axis=3)
    # if (undo):
    #     out_imgs = out_imgs.astype(np.uint8)
    return out_imgs

def Vgg_preprocess(imgs, undo=False):
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
    op_imgs = tf.concat(axis=3, values=channels)
    with tf.Session() as sess:
        out_imgs = sess.run(op_imgs)
    return out_imgs

class OfficialModel():
    root = '../official_data/'
    model = {
        'inception_v1': {
            'arg_scope': inception_v1.inception_v1_arg_scope,
            'graph': inception_v1.inception_v1,
            'checkpoint_dir': root + 'model/inception_v1/inception_v1.ckpt',
            'preprocess': Inception_preprocess,
            'min_max': (-1.0, 1.0)
        },
        'resnetv1_50': {
            'arg_scope': resnet_v1.resnet_arg_scope,
            'graph': resnet_v1.resnet_v1_50,
            'checkpoint_dir': root + 'model/resnet_v1_50/model.ckpt-49800',
            'preprocess': Vgg_preprocess_np,
            'min_max': (0, 255)
        },
        'vgg_16': {
            'arg_scope': vgg.vgg_arg_scope,
            'graph': vgg.vgg_16,
            'checkpoint_dir': root + 'model/vgg_16/vgg_16.ckpt',
            'preprocess': Vgg_preprocess_np,
            'min_max': (0, 255)
        }
    }

    def __init__(self, name):
        self.name = name
        self._model = OfficialModel.model[name]
    def load_model(self, x, nb_classes, reuse=None):
        with slim.arg_scope(self._model['arg_scope']()):
            logits, end_points = self._model['graph'](x, num_classes=nb_classes,is_training=False)
            if 'Logits' not in end_points:
                end_points['Logits'] = logits
            if (len(end_points['Logits'].shape) == 4):
                end_points['Logits'] = tf.squeeze(end_points['Logits'] , [1,2])
            if 'Predictions' not in end_points:
                end_points['Predictions'] = layers_lib.softmax(end_points['Logits'], scope='predictions')
            
        return end_points
    def get_weight(self):
        return self._model['checkpoint_dir']
    def preprocess(self, imgs):
        return self._model['preprocess'](imgs)
    def undo_preprocess(self, imgs):
        return self._model['preprocess'](imgs, undo=True)




def load_model_old(name, x_input, nb_classes, reuse=None):
    if (name == 'inception_v1'):
        with slim.arg_scope(inception_v1.inception_v1_arg_scope()):
            _, end_points = inception_v1.inception_v1(
                    x_input, num_classes=nb_classes, is_training=False, reuse=reuse)
    elif (name == 'resnetv1_50'):
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            _, end_points = resnet_v1.resnet_v1_50(x_input, num_classes=nb_classes,is_training=False, reuse=reuse)
            end_points['Logits'] = tf.squeeze(end_points['Logits'] , [1,2])
    return end_points

def weight_path(name):
    if (name == 'inception_v1'):
            return 'model/inception_v1/inception_v1.ckpt'
    elif (name == 'resnetv1_50'):
            return 'model/resnet_v1_50/model.ckpt-49800'

def preprocess_input(name, imgs):
    if (name == 'inception_v1'):
        return inception_v3.preprocess_input(imgs)
    elif (name == 'resnetv1_50'):
        return resnet50.preprocess_input(imgs)
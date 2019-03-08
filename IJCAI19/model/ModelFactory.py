import tensorflow as tf
slim = tf.contrib.slim
import numpy as np
from tensorflow.contrib.slim.python.slim.nets import inception_v1, resnet_v1
from tensorflow.contrib.layers.python.layers import layers as layers_lib

from .GhostNet import ghost_resnet_v1, vgg

from ..module.utils import *
from ..module.utils_tf import *
from .KerasModel import *

class BaseModel():
    WEIGHT_DIR = ""
    def __init__(self, name, nb_classes):
        self.name = name
        self.nb_classes = nb_classes
        self.predict_graph = False
        self.sess = None
    def get_endpoints(self, x, nb_classes=None):
        with slim.arg_scope(self._model['arg_scope']()):
            # print("get_endpoints", self.name)
            logits, end_points = self._model['graph'](x, num_classes=self.nb_classes,is_training=False,reuse=tf.AUTO_REUSE)
            if 'Logits' not in end_points:
                end_points['Logits'] = logits
            if (len(end_points['Logits'].shape) == 4):
                end_points['Logits'] = tf.squeeze(end_points['Logits'] , [1,2])
            if 'Predictions' not in end_points:
                end_points['Predictions'] = layers_lib.softmax(end_points['Logits'], scope='predictions')
        return end_points
    def _get_weight(self):
        return BaseModel.WEIGHT_DIR + self._model['checkpoint_dir']
    def load_weight(self, checkpoint_path=''):
        # if self.weight_loaded == False:
        if True:
            saver = tf.train.Saver(slim.get_model_variables(scope=self._model['var_scope']))
            if checkpoint_path == '':
                checkpoint_path = self._get_weight()
            sess = self.sess
            # sess = tf.get_default_session()
            saver.restore(sess, checkpoint_path)
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
        imgs = self._input_resize(imgs)
        return self._model['preprocess'](imgs)
    def clear_session(self):
        if self.sess:
            self.sess.close()
        self.sess = None
        tf.reset_default_graph()
    def predict_create_graph(self, batch_shape, use_prob=False, TOP_K=1):
        if self.sess:
            self.clear_session()
        config = gpu_session_config()
        self.sess =  tf.Session(config=config)
        self.x = tf.placeholder(dtype=tf.float32, shape=(None, batch_shape[1],batch_shape[2],batch_shape[3]), name='input')
        self.y = tf.placeholder(dtype=tf.float32, shape=(None, self.nb_classes), name='output')
        x = self.preprocess(self.x)
        if use_prob :
            op_prob = self.get_endpoints(x, self.nb_classes)['Predictions'].op.inputs[0]
            self.op_topk = tf.nn.in_top_k(op_prob, tf.argmax(self.y, 1), TOP_K)
            self.op_ypred = op_prob
        else:
            op_logits = self.get_endpoints(x, self.nb_classes)['Logits']
            self.op_topk = tf.nn.in_top_k(op_logits, tf.argmax(self.y, 1), TOP_K)
            self.op_ypred = tf.argmax(op_logits, 1)
        self.op_accuracy = tf.reduce_mean(tf.cast(self.op_topk, tf.float32))
        self.load_weight()
        return self.op_ypred
    def predict_batch(self, X, Y=None):
        if Y is not None:
            ypred, topk, accuracy= self.sess.run([self.op_ypred, self.op_topk, self.op_accuracy], feed_dict={self.x: X, self.y: Y})
            return ypred, topk, accuracy
        else:
            ypred = self.sess.run(self.op_ypred, feed_dict={self.x: X})
            return ypred 
    def predict_generator(self, generator, batch_shape, use_prob=False):
        p=Profile('Predict ')
        total_ypred = []
        self.predict_create_graph(batch_shape, use_prob)
        for _,X,Y in generator:
            ypred = self.predict_batch(X, None)
            total_ypred += [ypred]
        p.stop()
        return np.concatenate(total_ypred)
    def evaluate_generator(self, generator, batch_shape, use_prob=False):
        batch_iter = 0
        total_correct = 0
        total_size = 0
        total_ypred =None
        total_topk = None
        p=Profile('Predict ')
        self.predict_create_graph(batch_shape, use_prob)
        for _,X,Y in generator:
            batch_size =  X.shape[0]
            if Y is not None:
                ypred, topk, accuracy = self.predict_batch(X, Y)
                total_correct += (accuracy*batch_size)
                if total_topk is None:
                    total_topk = topk
                else:   
                    total_topk = np.concatenate((total_topk, topk), axis=0)
            else:
                ypred = self.predict_batch(X, None)

            batch_iter +=1
            total_size += batch_size
            if total_ypred is None:
                total_ypred = ypred
            else:   
                total_ypred = np.concatenate((total_ypred, ypred), axis=0)
            print(ypred.shape, total_ypred.shape)
        p.stop()

        if total_topk is not None:
            total_accuracy = total_correct/total_size
            print("batchs {0} Accuracy {1}".format(batch_iter, total_accuracy))
        return total_ypred, total_topk, total_accuracy



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


class factory_inception_v1(BaseModel):
    def __init__(self, *arg, **kwargs):
        super().__init__(*arg, **kwargs)
        self._model ={
            'var_scope': 'InceptionV1',
            'arg_scope': inception_v1.inception_v1_arg_scope,
            'graph': inception_v1.inception_v1,
            'checkpoint_dir':  'inception_v1/inception_v1.ckpt',
            'preprocess': Inception_preprocess,
            'default_input_size': 224,
        }
class factory_resnetv1_50(BaseModel):
    def __init__(self, *arg, **kwargs):
        super().__init__(*arg, **kwargs)
        self._model ={
            'var_scope': 'resnet_v1_50',
            'arg_scope': resnet_v1.resnet_arg_scope,
            'graph': resnet_v1.resnet_v1_50,
            'checkpoint_dir':  'resnet_v1_50/model.ckpt-49800',
            'preprocess': Vgg_preprocess_tf,
            'default_input_size': 224,
        }
class factory_vgg_16(BaseModel):
    def __init__(self, *arg, **kwargs):
        super().__init__(*arg, **kwargs)
        self._model ={
            'var_scope': 'vgg_16',
            'arg_scope': vgg.vgg_arg_scope,
            'graph': vgg.vgg_16,
            'checkpoint_dir':  'vgg_16/vgg_16.ckpt',
            'preprocess': Vgg_preprocess_tf,
            'default_input_size': 224,
        }
class factory_ghost_resnetv1_50(factory_resnetv1_50):
    def __init__(self, *arg, **kwargs):
        super().__init__(*arg, **kwargs)
        self._model['arg_scope'] = ghost_resnet_v1.resnet_arg_scope
        self._model['graph'] = ghost_resnet_v1.resnet_v1_50




class ModelFactory():
    WEIGHT_DIR = '../weight/'
    mapping = {
        'inception_v1': factory_inception_v1,
        'resnetv1_50': factory_resnetv1_50,
        'vgg_16': factory_vgg_16,
        'ghost_resnetv1_50': factory_ghost_resnetv1_50,
        'keras_xception_19': factory_keras_xception_19
    }
    def __init__(self):
        pass
    @staticmethod
    def create(name, nb_classes):
        KerasModel.WEIGHT_DIR = ModelFactory.WEIGHT_DIR
        BaseModel.WEIGHT_DIR = ModelFactory.WEIGHT_DIR
        return ModelFactory.mapping[name](name, nb_classes)
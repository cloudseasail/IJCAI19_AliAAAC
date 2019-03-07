import tensorflow as tf
import numpy as np
import os
from keras.models import load_model

def unify_preprocess(imgs, undo=False):
    if (undo):
        out_imgs = (((imgs + 1.0) * 0.5) * 255.0)
    else:
        out_imgs = (imgs / 255.0) * 2.0 - 1.0
    return out_imgs
    
class KerasModel():
    WEIGHT_DIR = ""
    def __init__(self, name):
        self.name = name
        self._model ={
            'preprocess': unify_preprocess,
            'default_input_size': 299,

        }
        self.model = None
    def get_endpoints(self, x, nb_classes):
       return None
    def load_weight(self, checkpoint_path=''):
       pass
    def _load_model(self, path):
        if os.path.exists(path):
            self.model = load_model(path)
            print("loaded keras model from ", path)
        else:
            print("keras model path not exit", path)
        return self.model
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
    def predict_create_graph(self, sess, batch_shape, use_prob=False, TOP_K=1):
        pass
    def predict_batch(self, X, Y=None):
        X = self.preprocess(X)
        ypred = self.model.predict_on_batch(X)
        ypred = ypred.argmax(1)
        if Y is not None:
            return ypred, None, None
        else:
            return ypred
    def evaluate_generator(self, generator, use_prob=False):
        total_ypred = []
        total_correct = 0
        total_size = 0
        for _,X,Y in generator:
            ypred = self.predict_batch(X)
            total_ypred = total_ypred+[ypred]
            total_correct += X[ypred == Y.argmax(1)].shape[0]
            total_size+= X.shape[0]
        total_accuracy = total_correct/total_size
        print("size {0}, Accuracy {0}".format(total_size, total_accuracy))
        return np.concatenate(total_ypred), None, total_accuracy


class factory_keras_xception_19(KerasModel):
    def __init__(self, name, nb_classes):
        super().__init__(name)
        path = KerasModel.WEIGHT_DIR + 'xception_19/keras_xception_19.h5'
        self.model = self._load_model(path)
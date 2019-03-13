import tensorflow as tf
import numpy as np
import os
from keras.models import load_model
import keras.backend as K
from cleverhans.utils_keras import KerasModelWrapper

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
        self.cleverhans_model = None
    def load_weight(self, sess=None, checkpoint_path=''):
        if self.model is None:
            self._load_model(self.weight_path)
        else:
            self.model.load_weights(self.weight_path)
            print("loaded keras model weights from ", self.weight_path)
    def _load_model(self, path):
        if os.path.exists(path):
            self.model = load_model(path)
            self.cleverhans_model = KerasModelWrapper(self.model)
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
    def predict_create_graph(self, batch_shape, use_prob=False, TOP_K=1):
        if (use_prob == False):
            print("Keras Model,  use_prob==False not implemented!!")
        if self.sess:
            self.clear_session()
        config = gpu_session_config()
        self.sess =  tf.Session(config=config)
        with self.sess.as_default():
            self.load_weight(self.sess)
    def predict_batch(self, X, Y=None):
        with self.sess.as_default():
            X = self.preprocess(X)
            ypred = self.model.predict_on_batch(X)
        # ypred = ypred.argmax(1)
        if Y is not None:
            return ypred, None, None
        else:
            return ypred
    def evaluate_generator(self, generator, batch_shape=None, use_prob=True):
        total_ypred = []
        total_correct = 0
        total_size = 0
        for _,X,Y in generator:
            ypred = self.predict_batch(X)
            total_ypred = total_ypred+[ypred]
            total_correct += X[ypred.argmax(1) == Y.argmax(1)].shape[0]
            total_size+= X.shape[0]
            # print(total_correct, total_size)
        total_accuracy = float(total_correct/total_size)
        return np.concatenate(total_ypred), None, total_accuracy
    def clear_session(self):
        if self.sess:
            self.sess.close()
        self.sess = None
        K.clear_session()
        del self.model
        self.model = None
        tf.reset_default_graph()
    def reload(self):
        self.clear_session()
        self.model = self._load_model(self.weight_path)
    def get_logits(self, x, nb_classes):
        return self.cleverhans_model.get_logits(x)
    def get_probs(self, x, nb_classes):
        return self.cleverhans_model.get_probs(x)


class factory_keras_xception_19(KerasModel):
    def __init__(self, name, nb_classes):
        super().__init__(name)
        self.weight_path = KerasModel.WEIGHT_DIR + 'xception_19/keras_xception_19.h5'
        #use lazy load weight
        # self.model = self._load_model(self.weight_path)

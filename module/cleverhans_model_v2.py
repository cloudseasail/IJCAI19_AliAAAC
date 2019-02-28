import tensorflow as tf
slim = tf.contrib.slim
from cleverhans.attacks import Model
from .official_model_v2 import OfficialModel,Inception_preprocess
from .utils import *

class CleverhansModel(Model):
    """Model class for CleverHans library."""

    def __init__(self, nb_classes, model=None):
        super(CleverhansModel, self).__init__(nb_classes=110)
        self.built = False
        end_points = None
        self._model = model

    def fprop(self, x, **kwargs):
        reuse = True if self.built else None
        end_points = self._model.load_model(x, self.nb_classes, reuse=reuse)
        self.built = True
        return {self.O_LOGITS: end_points['Logits'],
              self.O_PROBS: end_points['Predictions'].op.inputs[0]}


        

class AttackModel():
    def __init__(self, batch_shape=None, output_size=None, name=''):
        self.graph = tf.Graph()
        self.weight_loaded = False
        config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        config.gpu_options.allow_growth = True
        self.sess =  tf.Session(graph=self.graph, config=config)
        if batch_shape:
            with self.sess.as_default():
                with self.sess.graph.as_default():
                    x = tf.placeholder(dtype=tf.float32, shape=(None, batch_shape[1],batch_shape[2],batch_shape[3]), name='input')
                    y = tf.placeholder(dtype=tf.float32, shape=(None, output_size), name='output')
                    self.setting(x,y,name)
                    self._model = OfficialModel(name)
                    self.target_model = CleverhansModel(self.nb_classes, self._model)
    def setting(self, x, y, name=''):
        self.x = x
        self.y = y
        self.batch_size = x.shape[0]
        self.input_size = x.shape[2]
        self.output_size = y.shape[1]
        self.nb_classes = self.output_size
        self.name = name
    def load(self, name=''):
        if name:
            self.name = name
        self._model = OfficialModel(name)
        # with self.sess.as_default():
        #     with self.sess.graph.as_default():
        #         self.load_model()
        #         self.load_weight()

    def _load_weight(self, checkpoint_path=''):
        if self.weight_loaded == False:
            saver = tf.train.Saver(slim.get_model_variables())
            if checkpoint_path == '':
                checkpoint_path = self._model.get_weight()
            saver.restore(self.sess, checkpoint_path)
            self.weight_loaded = True
    def preprocess_input(self, imgs):
        return self._model.preprocess(imgs)
    def undo_preprocess(self, imgs):
        return self._model.undo_preprocess(imgs)
    def _predict(self, X, Y, TOP_K=1):
        # X = self.preprocess_input(X)
        with self.sess.as_default():
            with self.sess.graph.as_default():
                op_logits = self.target_model.get_logits(self.x)
                op_accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(op_logits, tf.argmax(self.y, 1), TOP_K), tf.float32))
                op_ypred = tf.argmax(op_logits, 1)
                self._load_weight()
                ypred, accuracy= self.sess.run([op_ypred, op_accuracy], feed_dict={self.x: X, self.y: Y})
                #out, end_points2, accuracy= sess.run([op_logits, self.op_end_points, op_accuracy], feed_dict={self.imgs_holder: X, self.labels_holder: Y})
        return ypred, accuracy

    def _attack(self, X, Y, Method, params):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                attacker = Method(self.target_model, sess=self.sess)
                op_xadv = attacker.generate(self.x, **params)
                self._load_weight()
                xadv = self.sess.run(op_xadv, feed_dict={self.x: X, self.y: Y})
                # op_accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(op_logits, tf.argmax(self.y, 1), 1), tf.float32))
                # op_ypred = tf.argmax(op_logits, 1)
                # ypred, accuracy= self.sess.run([op_ypred, op_accuracy], feed_dict={self.x: xadv, self.y: Y})
        return xadv

        

class Attacker(AttackModel):
    def __init__(self, batch_shape=None, output_size=None, name=''):
        super(Attacker, self).__init__(batch_shape, output_size, name)
    def _attack_preprocess(self, X, undo=False):
        if self.preprocess is None:
            return self._preprocess_default(X, undo)
        elif self.preprocess == '(-1,1)':
            return self._preprocess_xception(X, undo)
        else:
            pass
    # def load(self):
    #     with self.sess.as_default():
    #         with self.sess.graph.as_default():
    #             self._load_weight()
    def predict(self, X, Y):
        p=Profile(self.name+' predict')
        X_ = self.preprocess_input(X)
        ypred,accuracy = self._predict(X_, Y)
        # X = self.undo_preprocess(X_)
        p.stop()
        print("{0} predict accuracy : {1}".format(self.name, accuracy))
        return ypred,accuracy

    def attack(self,  X, Y, Method, params, preprocess=None):
        self.preprocess = preprocess
        p=Profile(self.name+' attack')
        X_, _min, _max = self._attack_preprocess(X)
        params['clip_min'] = _min
        params['clip_max'] = _max
        Xadv = self._attack(X_, Y, Method, params)
        # X,_,_ = self._attack_preprocess(X_, undo=True)
        Xadv,_,_ = self._attack_preprocess(Xadv, undo=True)
        p.stop()

        ypred, accuracy = self.predict(Xadv, Y)
        print("{0} adv accuracy : {1}, L2 {2}".format(self.name, accuracy, calc_l2(X, Xadv)))
        return Xadv, ypred
    

    def _preprocess_default(self, X, undo=False):
        X = self._model._model['preprocess'](X, undo=undo)
        (_min, _max) = self._model._model['min_max']
        return X, _min, _max
    def _preprocess_xception(self, X, undo=False):
        X = Inception_preprocess(X, undo=undo)
        (_min, _max) = (-1.0, 1.0)
        return X, _min, _max
    def _preprocess_vgg(self, X, undo=False):
        pass

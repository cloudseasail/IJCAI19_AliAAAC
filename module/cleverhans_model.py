import tensorflow as tf
slim = tf.contrib.slim
from cleverhans.attacks import Model
from .official_model_v2 import OfficialModel
from .utils import calc_l2

class CleverhansModel(Model):
    """Model class for CleverHans library."""

    def __init__(self):
        super(CleverhansModel, self).__init__(needs_dummy_fprop=True)
        self.built = False

    def get_logits(self,x=None):
        return self.end_points['Logits']

    def get_probs(self,x=None):
        return self.end_points['Predictions'].op.inputs[0]

    def fprop(self, x, **kwargs):
        return {self.O_LOGITS: self.get_logits(),
              self.O_PROBS: self.get_probs()}


        

class AttackModel(CleverhansModel):
    def __init__(self, batch_shape=None, output_size=None, name=''):
        super(AttackModel, self).__init__()
        self.graph = tf.Graph()
        self.sess =  tf.Session(graph=self.graph, config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
        # self.sess =  tf.Session(graph=self.graph)
        if batch_shape:
            with self.sess.as_default():
                with self.sess.graph.as_default():
                    x = tf.placeholder(dtype=tf.float32, shape=batch_shape, name='input')
                    y = tf.placeholder(dtype=tf.float32, shape=(None, output_size), name='output')
                    self.setting(x,y,name)
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
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.load_model()
                self.load_weight()
    def load_model(self):
        self.end_points = self._model.load_model(self.x, self.nb_classes)
        self.op_logits = self.get_logits()
        self.op_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=self.op_logits, onehot_labels=self.y))
    def load_weight(self, checkpoint_path=''):
        saver = tf.train.Saver(slim.get_model_variables())
        if checkpoint_path == '':
            checkpoint_path = self._model.get_weight()
        saver.restore(self.sess, checkpoint_path)
    def preprocess_input(self, imgs):
        return self._model.preprocess(imgs)
    def predict(self, X, Y, TOP_K=1):
        # X = self.preprocess_input(X)
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.op_accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.op_logits, tf.argmax(self.y, 1), TOP_K), tf.float32))
                op_ypred = tf.argmax(self.op_logits, 1)
                ypred, accuracy= self.sess.run([op_ypred, self.op_accuracy], feed_dict={self.x: X, self.y: Y})
                #out, end_points2, accuracy= sess.run([self.op_logits, self.op_end_points, op_accuracy], feed_dict={self.imgs_holder: X, self.labels_holder: Y})
        return ypred, accuracy

    def attack(self, X, Y, Method, params):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                attacker = Method(self, sess=self.sess)
                op_xadv = attacker.generate(self.x, **params)
                xadv = self.sess.run(op_xadv, feed_dict={self.x: X, self.y: Y})
                op_accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.op_logits, tf.argmax(self.y, 1), 1), tf.float32))
                op_ypred = tf.argmax(self.op_logits, 1)
                ypred, accuracy= self.sess.run([op_ypred, op_accuracy], feed_dict={self.x: xadv, self.y: Y})
        return xadv, ypred, accuracy

    def L2(self, X, Xadv):
        X_ = self._model.undo_preprocess(X)
        Xadv_ = self._model.undo_preprocess(Xadv)
        return calc_l2(X, Xadv)
        


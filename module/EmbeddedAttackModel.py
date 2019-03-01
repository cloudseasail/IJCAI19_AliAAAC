import tensorflow as tf
from cleverhans.attacks import Model
from .utils import *
from .utils_tf import *
from .official_model_v3 import OfficialModel

class CleverhansModel(Model):
    def __init__(self, nb_classes):
        super(CleverhansModel, self).__init__(nb_classes=110)
        self.built = False
        end_points = None

    def fprop(self, x, **kwargs):
        reuse = True if self.built else None
        logits = self.embedded_logits(x, self.nb_classes, reuse=reuse)
        self.built = True
        return {self.O_LOGITS: logits,
              self.O_PROBS: None}

class EmbeddedAttackModel(CleverhansModel):
    def __init__(self, batch_shape=None, output_size=None, name=''):
        super(EmbeddedAttackModel, self).__init__(nb_classes=output_size)
        config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        config.gpu_options.allow_growth = True
        graph = tf.Graph()
        # self.name =name
        self.sess =  tf.Session(graph=graph, config=config)
        self.models = []
        self.batch_shape = batch_shape
        with self.sess.as_default():
            with self.sess.graph.as_default():   
                self.op_x = tf.placeholder(dtype=tf.float32, shape=(None, self.batch_shape[1],self.batch_shape[2],self.batch_shape[3]), name='input')
                self.op_y = tf.placeholder(dtype=tf.float32, shape=(None, self.nb_classes), name='output')
    def add_model(self, models=[]):
        self.models.append(models)
    def embedded_logits(self, x, nb_classes, reuse=None):
        # embedded_logits = tf.zeros(shape=(None, nb_classes))
        embedded_logits = None
        for m in self.models:
            logits = m.get_attack_logits(x)
            # print("embedded_logits", logits)
            if embedded_logits is None:
                embedded_logits = logits
            else:
                embedded_logits += logits
        return embedded_logits/len(self.models)
    def attack_generate(self, Method, params):
        params = self.parse_attack_params(params)
        with self.sess.as_default():
            with self.sess.graph.as_default():
                x = tf.placeholder(dtype=tf.float32, shape=(None, self.batch_shape[1],self.batch_shape[2],self.batch_shape[3]), name='input')
                y = tf.placeholder(dtype=tf.float32, shape=(None, self.nb_classes), name='output')
                attacker = Method(self, sess=self.sess)
                self.op_xadv = attacker.generate(self.op_x, **params)
                for m in self.models:
                    m.load_weight()
        return self.op_xadv 
    def attack_batch(self, X, Y=None):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                xadv = self.sess.run(self.op_xadv, feed_dict={self.op_x: X, self.op_y: Y})
        return xadv
    def attack(self, X, Y, Method, params):
        params = self.parse_attack_params(params)
        # p=Profile(self.name+' predict')
        with self.sess.as_default():
            with self.sess.graph.as_default():
                attacker = Method(self, sess=self.sess)
                self.op_xadv = attacker.generate(self.op_x, **params)
                for m in self.models:
                    m.load_weight()
                xadv = self.sess.run(self.op_xadv, feed_dict={self.op_x: X, self.op_y: Y})
        # p.stop()
        return xadv
    def parse_attack_params(self, params):
        (_min, _max) = (0,255)
        if "ep_ratio" in params:
            params['eps'] = (_max-_min)*params['ep_ratio']
            params['eps_iter'] = params['eps']/10
        if "nb_iter" in params: 
            params['eps_iter'] = params['eps']/params['nb_iter']
        params['clip_min'] = _min
        params['clip_max'] = _max
        print(params)
        return params



class TargetModel(OfficialModel):
    def __init__(self, batch_shape=None, output_size=None, name=''):
        super(TargetModel, self).__init__(name=name)
        graph = tf.Graph()
        self.name = name
        self.nb_classes = output_size
        config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        config.gpu_options.allow_growth = True
        self.sess =  tf.Session(graph=graph, config=config)
        self.models = []
        self.batch_shape = batch_shape
        if batch_shape:
            with self.sess.as_default():
                with self.sess.graph.as_default():
                    self.x = tf.placeholder(dtype=tf.float32, shape=(None, batch_shape[1],batch_shape[2],batch_shape[3]), name='input')
                    self.y = tf.placeholder(dtype=tf.float32, shape=(None, output_size), name='output')
    def predict_generate(self, TOP_K=1):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                x = self.predict_preprocess(self.x)
                op_logits = self.get_endpoints(x, self.nb_classes)['Logits']
                self.op_accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(op_logits, tf.argmax(self.y, 1), TOP_K), tf.float32))
                self.op_ypred = tf.argmax(op_logits, 1)
                self.load_weight()
        return self.op_ypred
    def predict_batch(self, X, Y):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                ypred, accuracy= self.sess.run([self.op_ypred, self.op_accuracy], feed_dict={self.x: X, self.y: Y})
        return ypred, accuracy
    def predict(self, X, Y, TOP_K=1):
        p=Profile(self.name+' predict')
        with self.sess.as_default():
            with self.sess.graph.as_default():
                x = self.predict_preprocess(self.x)
                op_logits = self.get_endpoints(x, self.nb_classes)['Logits']
                op_accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(op_logits, tf.argmax(self.y, 1), TOP_K), tf.float32))
                op_ypred = tf.argmax(op_logits, 1)
                self.load_weight()
                ypred, accuracy= self.sess.run([op_ypred, op_accuracy], feed_dict={self.x: X, self.y: Y})
        p.stop()
        print("{0} predict accuracy : {1}".format(self.name, accuracy))
        return ypred, accuracy
    def get_attack_logits(self, x):
        xp = self.attack_preprocess(x)
        logits = self.get_endpoints(xp,self.nb_classes)['Logits']
        logits = grad_norm(logits)
        return logits
    def get_weight_path(self):
        return None


def AttackHelper(A, X, Y, M, param):
    p=Profile('Attack')
    A.attack_generate(M, param)
    Xadv = A.attack_batch(X, Y)
    p.stop()
    print("L2", calc_l2(X, Xadv))
    for m in A.models:
        m.predict(Xadv, Y)
    return Xadv

def AttackBatch(A, gen, M, param, max_iter=None):
    # batch_size = A.batch_shape[0]
    batch_iter = 0
    total_l2 = 0
    total_size = 0
    A.attack_generate(M, param)
    p=Profile('Attack ')
    for _,X,Y in gen:
        Xadv = A.attack_batch(X, Y)
        batch_iter +=1
        l2 = calc_l2(X, Xadv)
        batch_size =  X.shape[0]
        total_l2 += (l2*batch_size)
        total_size += batch_size
        if (max_iter):
            if batch_iter>=max_iter:
                break
    p.stop()
    total_l2 = total_l2/total_size
    print("batchs {0} L2 {1}".format(batch_iter, total_l2))
    return Xadv


def PredictBatch(T, gen):
    # batch_size = T.batch_shape[0]
    batch_iter = 0
    total_correct = 0
    total_size = 0
    T.predict_generate()
    p=Profile('Predict ')
    for _,X,Y in gen:
        ypred, accuracy = T.predict_batch(X, Y)
        batch_iter +=1
        batch_size =  X.shape[0]
        total_correct += (accuracy*batch_size)
        total_size += batch_size
        # print(batch_size, total_correct, total_size)
    p.stop()
    total_accuracy = total_correct/total_size
    print("batchs {0} Accuracy {1}".format(batch_iter, total_accuracy))
    with T.sess.as_default():
        with T.sess.graph.as_default():
            tf.reset_default_graph()
    return total_accuracy

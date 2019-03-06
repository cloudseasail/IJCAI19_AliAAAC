import tensorflow as tf
from cleverhans.attacks import Model
from .ModelFactory import ModelFactory
from .BatchModel import BatchModel
# import sys
# sys.path.append("../")
from ..module.utils import *
from ..module.utils_tf import *

class CleverhansModel(Model):
    def __init__(self, nb_classes):
        super(CleverhansModel, self).__init__(nb_classes=110)
        self.built = False
        end_points = None

    def fprop(self, x, **kwargs):
        # reuse = True if self.built else None
        logits = self.embedded_logits(x, self.nb_classes)
        probs = self.embedded_probs(x, self.nb_classes)
        self.built = True
        return {self.O_LOGITS: logits,
              self.O_PROBS: probs}

class EmbeddedAttackModel(CleverhansModel):
    def __init__(self, batch_shape=None, output_size=None, name=''):
        super(EmbeddedAttackModel, self).__init__(nb_classes=output_size)
        self.models = []
        self.batch_shape = batch_shape
        # with self.sess.as_default():
        #     with self.sess.graph.as_default():   
        #         self.op_x = tf.placeholder(dtype=tf.float32, shape=(None, self.batch_shape[1],self.batch_shape[2],self.batch_shape[3]), name='input')
        #         self.op_y = tf.placeholder(dtype=tf.float32, shape=(None, self.nb_classes), name='output')
    def add_model(self, models=[]):
        self.models.append(models)
    def embedded_logits(self, x, nb_classes):
        embedded_logits = []
        for m in self.models:
            logits = m.get_attack_logits(x)
            logits = grad_norm(logits)
            embedded_logits.append(logits)
        return tf.reduce_mean(embedded_logits, axis=0)
    def embedded_probs(self, x, nb_classes):
        embedded_probs = []
        for m in self.models:
            probs = m.get_attack_probs(x)
            #no grad normalize for probs!!
            embedded_probs.append(probs)
        return tf.reduce_mean(embedded_probs, axis=0)

    def attack_generate(self, sess, Method, params):
        self.sess = sess
        self.op_x = tf.placeholder(dtype=tf.float32, shape=(None, self.batch_shape[1],self.batch_shape[2],self.batch_shape[3]), name='input')
        self.op_y = tf.placeholder(dtype=tf.float32, shape=(None, self.nb_classes), name='output')
        params = self.parse_attack_params(params, self.op_y)
        attacker = Method(self, sess=self.sess)
        self.op_xadv = attacker.generate(self.op_x, **params)
        for m in self.models:
            m.load_weight()
        return self.op_xadv 
    def attack_batch(self, X, Y=None):
        xadv = self.sess.run(self.op_xadv, feed_dict={self.op_x: X, self.op_y: Y})
        return xadv
    def parse_attack_params(self, params, op_y=None):
        (_min, _max) = (0,255)
        if "ep_ratio" in params:
            params['eps'] = (_max-_min)*params['ep_ratio']
            params['eps_iter'] = params['eps']/10
        if "nb_iter" in params: 
            params['eps_iter'] = params['eps']/params['nb_iter']
        if "target" in params:
            if params['target']:
                params['y_target'] = op_y
            else:
                params['y'] = op_y

        params['clip_min'] = _min
        params['clip_max'] = _max
        print(params)
        return params



class AttackModel(BatchModel):
    def __init__(self, batch_shape=None, output_size=None, name='', use_prob=False):
        BatchModel.__init__(self, batch_shape=batch_shape, output_size=output_size, name=name, use_prob=use_prob)
        self.name = name
        self.nb_classes = output_size
        self.models = []
        self.batch_shape = batch_shape
        self.use_prob = use_prob
        self.model = None
        if name:
            self.model = ModelFactory.get_by_name(name)
    def get_attack_logits(self, x):
        xp = self.attack_preprocess(x)
        logits = self.get_endpoints(xp,self.nb_classes)['Logits']
        return logits
    def get_attack_probs(self, x):
        xp = self.attack_preprocess(x)
        probs = self.get_endpoints(xp,self.nb_classes)['Predictions'].op.inputs[0]
        return probs
    def predict_preprocess(self, x):
        if self.model:
            return self.model.predict_preprocess(x)
    def attack_preprocess(self, x):
        if self.model:
            return self.model.attack_preprocess(x)
    def load_weight(self, *arg, **kwargs):
        if self.model:
            return self.model.load_weight(*arg, **kwargs)
    def get_endpoints(self, *arg, **kwargs):
        if self.model:
            return self.model.get_endpoints(*arg, **kwargs)



def AttackBatch(A, gen, M, param, max_iter=None):
    # batch_size = A.batch_shape[0]
    batch_iter = 0
    total_l2 = 0
    total_size = 0
    Xadv = None
    config = gpu_session_config()
    p=Profile('Attack ')
    with tf.Session(config=config) as sess:
        A.attack_generate(sess, M, param)
        for _,X,Y in gen:
            Xadv_ = A.attack_batch(X, Y)
            batch_iter +=1
            l2 = calc_l2(X, Xadv_)
            batch_size =  X.shape[0]
            total_l2 += (l2*batch_size)
            total_size += batch_size
            if Xadv is None:
                Xadv = Xadv_
            else:   
                Xadv = np.concatenate((Xadv, Xadv_), axis=0)
            if (max_iter):
                if batch_iter>=max_iter:
                    break
    p.stop()
    tf.reset_default_graph()
    total_l2 = total_l2/total_size
    print("batchs {0} L2 {1}".format(batch_iter, total_l2))
    print("GPU status", get_gpu_status())
    return Xadv

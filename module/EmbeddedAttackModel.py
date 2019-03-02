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
        self.models = []
        self.batch_shape = batch_shape
        # with self.sess.as_default():
        #     with self.sess.graph.as_default():   
        #         self.op_x = tf.placeholder(dtype=tf.float32, shape=(None, self.batch_shape[1],self.batch_shape[2],self.batch_shape[3]), name='input')
        #         self.op_y = tf.placeholder(dtype=tf.float32, shape=(None, self.nb_classes), name='output')
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
    def attack_generate(self, sess, Method, params):
        self.sess = sess
        self.op_x = tf.placeholder(dtype=tf.float32, shape=(None, self.batch_shape[1],self.batch_shape[2],self.batch_shape[3]), name='input')
        self.op_y = tf.placeholder(dtype=tf.float32, shape=(None, self.nb_classes), name='output')
        params = self.parse_attack_params(params)
        attacker = Method(self, sess=self.sess)
        self.op_xadv = attacker.generate(self.op_x, **params)
        for m in self.models:
            m.load_weight()
        return self.op_xadv 
    def attack_batch(self, X, Y=None):
        xadv = self.sess.run(self.op_xadv, feed_dict={self.op_x: X, self.op_y: Y})
        return xadv
    def parse_attack_params(self, params):
        (_min, _max) = (0,255)
        if "ep_ratio" in params:
            params['eps'] = (_max-_min)*params['ep_ratio']
            params['eps_iter'] = params['eps']/10
        if "nb_iter" in params: 
            params['eps_iter'] = params['eps']/params['nb_iter']
        if "y_target " not in params:
            params['y'] = self.op_y

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
        self.models = []
        self.batch_shape = batch_shape
    def predict_generate(self, sess, TOP_K=1):
        self.sess = sess
        self.x = tf.placeholder(dtype=tf.float32, shape=(None, self.batch_shape[1],self.batch_shape[2],self.batch_shape[3]), name='input')
        self.y = tf.placeholder(dtype=tf.float32, shape=(None, self.nb_classes), name='output')
        x = self.predict_preprocess(self.x)
        op_logits = self.get_endpoints(x, self.nb_classes)['Logits']
        self.op_accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(op_logits, tf.argmax(self.y, 1), TOP_K), tf.float32))
        self.op_ypred = tf.argmax(op_logits, 1)
        self.load_weight()
        return self.op_ypred
    def predict_batch(self, X, Y):
        ypred, accuracy= self.sess.run([self.op_ypred, self.op_accuracy], feed_dict={self.x: X, self.y: Y})
        return ypred, accuracy
    def get_attack_logits(self, x):
        xp = self.attack_preprocess(x)
        logits = self.get_endpoints(xp,self.nb_classes)['Logits']
        logits = grad_norm(logits)
        return logits
    def get_weight_path(self):
        return None



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


def PredictBatch(T, gen):
    # batch_size = T.batch_shape[0]
    batch_iter = 0
    total_correct = 0
    total_size = 0
    p=Profile('Predict ')
    config = gpu_session_config()
    with tf.Session(config=config) as sess:
        T.predict_generate(sess)
        for _,X,Y in gen:
            ypred, accuracy = T.predict_batch(X, Y)
            batch_iter +=1
            batch_size =  X.shape[0]
            total_correct += (accuracy*batch_size)
            total_size += batch_size
            # print(batch_size, total_correct, total_size)
    p.stop()
    tf.reset_default_graph()
    total_accuracy = total_correct/total_size
    print("batchs {0} Accuracy {1}".format(batch_iter, total_accuracy))
    
    return total_accuracy

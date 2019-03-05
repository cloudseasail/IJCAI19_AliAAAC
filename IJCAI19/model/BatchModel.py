import tensorflow as tf
from ..module.utils import *
from ..module.utils_tf import *

class BatchModel():
    def __init__(self, batch_shape=None, output_size=None, name='', use_prob=False):
        self.name = name
        self.nb_classes = output_size
        self.batch_shape = batch_shape
        self.use_prob = use_prob
    def predict_generate(self, sess, TOP_K=1):
        self.sess = sess
        self.x = tf.placeholder(dtype=tf.float32, shape=(None, self.batch_shape[1],self.batch_shape[2],self.batch_shape[3]), name='input')
        self.y = tf.placeholder(dtype=tf.float32, shape=(None, self.nb_classes), name='output')
        x = self.predict_preprocess(self.x)
        if self.use_prob :
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
    def predict_preprocess(self, x):
        print("BatchModel.predict_process not implemented")
        return x


def PredictBatch(T, gen):
    batch_iter = 0
    total_correct = 0
    total_size = 0
    total_ypred =None
    total_topk = None
    p=Profile('Predict ')
    config = gpu_session_config()
    with tf.Session(config=config) as sess:
        T.predict_generate(sess)
        for _,X,Y in gen:
            batch_size =  X.shape[0]
            if Y is not None:
                ypred, topk, accuracy = T.predict_batch(X, Y)
                total_correct += (accuracy*batch_size)
                if total_topk is None:
                    total_topk = topk
                else:   
                    total_topk = np.concatenate((total_topk, topk), axis=0)
            else:
                ypred = T.predict_batch(X, None)

            batch_iter +=1
            total_size += batch_size
            if total_ypred is None:
                total_ypred = ypred
            else:   
                total_ypred = np.concatenate((total_ypred, ypred), axis=0)
    p.stop()
    tf.reset_default_graph()

    if total_topk is not None:
        total_accuracy = total_correct/total_size
        print("batchs {0} Accuracy {1}".format(batch_iter, total_accuracy))
    
    return total_ypred, total_topk, total_accuracy
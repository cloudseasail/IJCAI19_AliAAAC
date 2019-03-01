import tensorflow as tf

def grad_norm(grad):
    avoid_zero_div = tf.cast(1e-12, grad.dtype)
    shape = get_flat_shape(grad)
    std = tf.reshape(tf.contrib.keras.backend.std(tf.reshape(grad, [shape[0], -1]), axis=1),shape)
    std = tf.maximum(avoid_zero_div, std)
    return grad/std
def get_flat_shape(grad):
    t = tf.zeros(shape=tf.shape(grad)[0])
    for i in range(1,len(grad.get_shape().as_list())):
        t= tf.expand_dims(t,i)
    return tf.shape(t)
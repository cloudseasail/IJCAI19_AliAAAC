import tensorflow as tf
class GhostMaker():
    def __init__(self, shortcut_weight_range=1.0, dropout_prob=0.0):
        # self.shortcut_weight = shortcut_weight
        self.dropout_prob = dropout_prob
        self.shortcut_weight_range = shortcut_weight_range 
    def get_shortcut_weight(self, depth):
        # print("shortcut_weight_range",  self.shortcut_weight_range)
        return tf.random_uniform((depth,), minval=1 - self.shortcut_weight_range, maxval=1 + self.shortcut_weight_range)
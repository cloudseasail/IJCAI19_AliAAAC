from keras.applications.xception import Xception, preprocess_input
from keras.applications.nasnet import NASNetLarge

from keras import models
from keras import layers

def XceptionFineTune19(weights=None, input_shape=None, num_classes=None):
    base_weright = None
    fine_weight = None
    if weights:
        if weights == 'imagenet':
            base_weright = weights
        else:
            fine_weight = weights
    xception = Xception(weights=base_weright, include_top=False, input_shape=input_shape)
    finetune_layer_after = "add_11"
    trainable = False
    for layer in xception.layers:
        if layer.name == finetune_layer_after:
            trainable = True
        layer.trainable = trainable

    model = models.Sequential() # 產生一個新的網絡模型結構
    model.add(xception)        # 把預訓練的卷積基底疊上去
    model.add(layers.GlobalAveragePooling2D(name='final_gap'))
    model.add(layers.Dense(num_classes,activation='softmax',kernel_initializer='he_normal', name='final_dense_mapping')) 

    if fine_weight:
        model.load_weights(fine_weight)
    return model


def NASNetLargeFineTune(weights=None, input_shape=None, num_classes=None):
    base_weright = None
    fine_weight = None
    if weights:
        if weights == 'imagenet':
            base_weright = weights
        else:
            fine_weight = weights
    basenet = NASNetLarge(weights=base_weright, include_top=False, input_shape=input_shape)
    finetune_layer_after = "normal_concat_16"  # 9%
    trainable = False
    for layer in basenet.layers:
        if layer.name == finetune_layer_after:
            trainable = True
        layer.trainable = trainable

    model = models.Sequential() # 產生一個新的網絡模型結構
    model.add(basenet)        # 把預訓練的卷積基底疊上去
    model.add(layers.GlobalAveragePooling2D(name='final_gap'))
    model.add(layers.Dense(num_classes,activation='softmax',kernel_initializer='he_normal', name='final_dense_mapping')) 

    if fine_weight:
        model.load_weights(fine_weight)
    return model
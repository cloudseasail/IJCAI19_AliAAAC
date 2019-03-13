from keras.applications.xception import Xception, preprocess_input
from keras.applications.nasnet import NASNetLarge

from keras.models import Model
from keras import layers

def finetune_xception_19(weights=None, input_shape=None, num_classes=None):
    base_weright = None
    fine_weight = None
    if weights:
        if weights == 'imagenet':
            base_weright = weights
        else:
            fine_weight = weights
    basenet = Xception(weights=base_weright, include_top=False, input_shape=input_shape)
    finetune_layer_after = "add_11"
    trainable = False
    for layer in basenet.layers:
        if layer.name == finetune_layer_after:
            trainable = True
        layer.trainable = trainable

    _input = basenet.input
    x = basenet.output
    x = layers.GlobalAveragePooling2D(name='final_gap')(x)
    _output = layers.Dense(num_classes,activation='softmax',kernel_initializer='he_normal', name='final_dense_mapping')(x)
    model = Model(inputs=_input, outputs=_output)

    if fine_weight:
        model.load_weights(fine_weight)
    return model

def finetune_nasnet_large(weights=None, input_shape=None, num_classes=None):
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

    _input = basenet.input
    x = basenet.output
    x = layers.GlobalAveragePooling2D(name='final_gap')(x)
    _output = layers.Dense(num_classes,activation='softmax',kernel_initializer='he_normal', name='final_dense_mapping')(x)
    model = Model(inputs=_input, outputs=_output)
    
    if fine_weight:
        model.load_weights(fine_weight)
    return model
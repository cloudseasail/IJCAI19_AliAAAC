from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import os
import sys
sys.path.append("..")
from module.EnhancedDataGenerator import MultiDataGenerator
from module.utils import * 
from module.utils_keras import  TensorBoardCallback
from model.xception_keras import XceptionFineTune19, preprocess_input

from keras import optimizers
from keras import callbacks
from keras.models import load_model
import keras.backend as K


IMAGE_SIZE = 299
CLASS_SIZE = 110
BATCH_SIZE = 26
VALIDATION_SPLIT = 0.05

####################################################
############## DataGenerator  ####################
####################################################
print("=========== Prepare Data Generator =====================")
sources = {
    'good':{
        'directory': '../../official_data/prepared_train_data/good/',
        'shuffle_num': 100,
    },
    'bad':{
        'directory': '../../official_data/prepared_train_data/bad/',
        'shuffle_num': 8,
    },
    'adv':{
        'directory': '../../official_data/prepared_train_data/adv/',
        'shuffle_num': 12,
    },
}
MDG = MultiDataGenerator(sources, 
                    msb_max=24, msb_rate=0.1, 
                    rotation_range=20,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    shear_range=0.1,
                    zoom_range=0.05,
                    channel_shift_range=4,
                    horizontal_flip=True,
                    fill_mode='nearest',
                    preprocessing_function = preprocess_input,
                    validation_split=VALIDATION_SPLIT)
train_generator = MDG.get_train_flow(target_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size = BATCH_SIZE)
validation_generator = MDG.get_valid_flow(target_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size = BATCH_SIZE)
tensorboard_generator = MDG.get_valid_flow(target_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size = BATCH_SIZE)
mutlgen_length = len(MDG)
print("MultiDataGenerator total length ", mutlgen_length)

TOTAL_SIZE = mutlgen_length*BATCH_SIZE
TRAIN_SIZE = int(TOTAL_SIZE * (1-VALIDATION_SPLIT))
EPOCH_SIZE = 10
STEPS_PRE_EPOCH = TRAIN_SIZE//BATCH_SIZE
print('TOTAL_SIZE {0} ,BATCH_SIZE {1}, STEPS_PRE_EPOCH {2},'.format(TOTAL_SIZE, BATCH_SIZE, STEPS_PRE_EPOCH))
#debug
STEPS_PRE_EPOCH = 1000
EPOCH_INIT = 0

################################################################
############## Model Symbolic: Transfer with fine tune  ########
################################################################
print("=========== Prepare Model =====================")
saved_model = 'model/keras_xception_19.h5'
if os.path.exists(saved_model):
    model = load_model(saved_model)
    print("loaded pretrained model from ", saved_model)
else:
    model = XceptionFineTune19('imagenet', (IMAGE_SIZE,IMAGE_SIZE,3), CLASS_SIZE)
    EPOCH_INIT = 0
model.summary()


################################################################
############## Training with params and callbacks    ########
################################################################
def _lr_schedule(epoch):
    lr = 1e-3
    if epoch > 10:
        lr *= 1e-2
    elif epoch > 1:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr
def lr_scheduler(epoch):
    lr = _lr_schedule(epoch)
    K.set_value(model.optimizer.lr, lr)
    print("lr changed from {0} to {1}".format(K.get_value(model.optimizer.lr), lr))
    return K.get_value(model.optimizer.lr)

model.compile(loss='categorical_crossentropy',
#               optimizer=optimizers.RMSprop(lr=2e-5),
              optimizer=optimizers.Adam(lr=_lr_schedule(EPOCH_INIT)),
              metrics=['accuracy'])

lrscheduler = callbacks.LearningRateScheduler(lr_scheduler)
lrreducer = callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, mode='auto', min_lr=0.5e-6)
tensorboard = TensorBoardCallback(tensorboard_generator, BATCH_SIZE, "logs/train/")
checkpointer = callbacks.ModelCheckpoint(filepath="logs/xception19.h5", verbose=1, save_best_only=True)
history = model.fit_generator(
      train_generator,
      steps_per_epoch=STEPS_PRE_EPOCH,
      epochs=EPOCH_SIZE,
      validation_data=validation_generator,
      validation_steps= max(STEPS_PRE_EPOCH*VALIDATION_SPLIT, 30),
#       use_multiprocessing=True,
      callbacks = [checkpointer, tensorboard, lrscheduler, lrreducer],
      verbose=1,
      initial_epoch=EPOCH_INIT)
      
model.save(saved_model) # 把模型儲存到檔案


################################################################
############## Training History   ########
################################################################
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, label='Training acc')
plt.plot(epochs, val_acc, label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, label='Training loss')
plt.plot(epochs, val_loss, label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
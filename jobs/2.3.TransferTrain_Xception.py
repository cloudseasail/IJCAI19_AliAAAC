from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import os
import sys
sys.path.append("..")
from IJCAI19.module.EnhancedDataGenerator import MultiDataGenerator
from IJCAI19.module.utils import * 
from IJCAI19.module.utils_keras import *
from IJCAI19.model.xception_keras import XceptionFineTune19, preprocess_input

from keras import optimizers
from keras import callbacks
from keras.models import load_model
import keras.backend as K
import csv

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
        'shuffle_num': 20,
    },
    'bad':{
        'directory': '../../official_data/prepared_train_data/bad/',
        'shuffle_num': 1,
    },
    'adv':{
        'directory': '../../official_data/prepared_train_data/adv/',
        'shuffle_num': 2,
    },
}
MDG = MultiDataGenerator(sources, 
                    msb_max=12, msb_rate=0.1, 
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
# tensorboard_generator = MDG.get_valid_flow(target_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size = BATCH_SIZE)
#tensorboard_generator has bug, disable 
tensorboard_generator=None
mutlgen_length = len(MDG)
print("MultiDataGenerator total length ", mutlgen_length)

TOTAL_SIZE = mutlgen_length*BATCH_SIZE
TRAIN_SIZE = int(TOTAL_SIZE * (1-VALIDATION_SPLIT))
EPOCH_SIZE = 100
STEPS_PRE_EPOCH = TRAIN_SIZE//BATCH_SIZE
print('TOTAL_SIZE {0} ,BATCH_SIZE {1}, STEPS_PRE_EPOCH {2},'.format(TOTAL_SIZE, BATCH_SIZE, STEPS_PRE_EPOCH))

#use smaller eoch size to save result more frequently
_STEPS_PRE_EPOCH = 200
_EPOCH_SIZE = int(EPOCH_SIZE * (STEPS_PRE_EPOCH/_STEPS_PRE_EPOCH))
EPOCH_INIT = 0
BEST_LOSS = None
print('Real _EPOCH_SIZE {0} ,EPOCH_INIT {1}, _STEPS_PRE_EPOCH {2},'.format(_EPOCH_SIZE, EPOCH_INIT, _STEPS_PRE_EPOCH))


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

saved_log = 'logs/keras_xception_19.csv'
if os.path.exists(saved_log):
    with open(saved_log) as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]
        EPOCH_INIT = int(rows[-1]['epoch'])+1
        BEST_LOSS = float(rows[-1]['val_loss'])
        print('Resume training from EPOCH_INIT {0} ,BEST_LOSS {1}, BEST_ACC {2}'.format(EPOCH_INIT, BEST_LOSS, rows[-1]['val_acc']))
################################################################
############## Training with params and callbacks    ########
################################################################
LR_SCHEULE_TABLE = {
    0: 1e-3,
    50: 1e-5,
    100: 5e-6,
    200: 1e-6
}
def _lr_schedule(epoch, old_lr=1e-3):
    if epoch in LR_SCHEULE_TABLE:
        lr = LR_SCHEULE_TABLE[epoch]
    else:
        lr = old_lr
    print("lr changed from {0} to {1}".format(old_lr, lr))
    return lr

#donot compile again to avoid optimizer breaking
if EPOCH_INIT == 0:
    model.compile(loss='categorical_crossentropy',
    #               optimizer=optimizers.RMSprop(lr=2e-5),
                optimizer=optimizers.Adam(lr=_lr_schedule(EPOCH_INIT)),
                metrics=['accuracy'])

lrscheduler = callbacks.LearningRateScheduler(_lr_schedule)
lrreducer = callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto', min_lr=1e-7)
tensorboard = TensorBoardCallback(tensorboard_generator, BATCH_SIZE, "logs/train/")
# checkpointer = callbacks.ModelCheckpoint(filepath=saved_model, verbose=1, save_best_only=True)
checkpointer = ModelCheckpointWrapper(best_init=BEST_LOSS, filepath=saved_model, verbose=1, save_best_only=True)
csvlogger = callbacks.CSVLogger(saved_log, separator=',', append=True)
history = model.fit_generator(
      train_generator,
      steps_per_epoch=_STEPS_PRE_EPOCH,
      epochs=_EPOCH_SIZE,
      validation_data=validation_generator,
      validation_steps= max(_STEPS_PRE_EPOCH*VALIDATION_SPLIT, 30),
#       use_multiprocessing=True,
      callbacks = [checkpointer, tensorboard, lrscheduler, lrreducer, csvlogger],
      verbose=1,
      initial_epoch=EPOCH_INIT)
      
# dont save fina mmodel,  becasuse best model already saved
# model.save(saved_model) # 把模型儲存到檔案


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
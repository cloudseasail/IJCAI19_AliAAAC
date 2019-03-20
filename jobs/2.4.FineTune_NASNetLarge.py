from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import os
import sys
sys.path.append("..")
from IJCAI19.module.EnhancedDataGenerator import MultiDataGenerator
from IJCAI19.module.utils import * 
from IJCAI19.module.utils_keras import *
from IJCAI19.model.KerasModelDef import finetune_nasnet_large, preprocess_input

from keras import optimizers
from keras import callbacks
from keras.models import load_model
import keras.backend as K
import csv

IMAGE_SIZE = 299
CLASS_SIZE = 110
BATCH_SIZE = 20
VALIDATION_SPLIT = 0

####################################################
############## DataGenerator  ####################
####################################################
print("=========== Prepare Data Generator =====================")
sources = {
    'good':{
        'directory': '../../official_data/prepared_train_data/good/',
        'shuffle_num': 20,
    },
    'adv':{
        'directory': '../../official_data/prepared_train_data/adv/',
        'shuffle_num': 1,
    },
    'adv2':{
        'directory': '../../official_data/prepared_train_data/adv2/',
        'shuffle_num': 1,
    },
}
MDG = MultiDataGenerator(sources,  source_names=['adv', 'good', 'adv2'],
                    msb_max=10, msb_rate=0.1, 
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

class ValidGenerator():
    def __init__(self, path):
        self.path = path
        self.batch_shape = [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3]
        self.len = 110
    def _regen(self):
        self.gen =  ImageLoader(self.path, self.batch_shape, targetlabel=False, label_size=CLASS_SIZE, format='png', label_file='dev.csv')
    def __len__(self):
        return self.len
    def _next(self):
        try:
            _,X,Y = next(self.gen)
        except:
            self._regen()
            _,X,Y = next(self.gen)
        if X.shape[0] < BATCH_SIZE:
            self._regen()
            _,X,Y = next(self.gen)
        return X,Y
    def __next__(self):
        X,Y = self._next()
        X= preprocess_input(X)
        return X,Y
#use 110 attcked adv dev_data for validaton, to make sure every epoch has same metric
adv_dev_dir= "\\\\rfsw-bj3\\Develop\\IJCAI2019\\test_data\\NonTargetAttackValidate"
validation_generator = ValidGenerator(adv_dev_dir)
VALID_STEPS = len(validation_generator)//BATCH_SIZE

mutlgen_length = len(MDG)
print("MultiDataGenerator total length {0},  VALID_STEPS {1}".format(mutlgen_length, VALID_STEPS))

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
saved_model = 'model/keras_nasnet_large.h5'
logged_model = 'logs/keras_nasnet_large.h5'
if os.path.exists(saved_model):
    model = load_model(saved_model)
    print("loaded pretrained model from ", saved_model)
else:
    model = finetune_nasnet_large('imagenet', (IMAGE_SIZE,IMAGE_SIZE,3), CLASS_SIZE)
    EPOCH_INIT = 0
model.summary()

saved_log = 'logs/keras_nasnet_large.csv'
if os.path.exists(saved_log):
    with open(saved_log) as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]
        if len(rows) > 0:
            EPOCH_INIT = int(rows[-1]['epoch'])+1
            BEST_LOSS = float(rows[-1]['val_loss'])
            print('Resume training from EPOCH_INIT {0} ,BEST_LOSS {1}, BEST_ACC {2}'.format(EPOCH_INIT, BEST_LOSS, rows[-1]['val_acc']))
################################################################
############## Training with params and callbacks    ########
################################################################
LR_SCHEULE_TABLE = {
    0: 1e-3,
    10: 1e-4,
    30: 1e-5,
    50: 5e-6,
    80: 1e-6,
    100: 1e-7
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
lrreducer = callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto', min_lr=1e-8)
tensorboard = TensorBoardCallback(None, BATCH_SIZE, "logs/train/")
checkpointer = ModelCheckpointWrapper(best_init=BEST_LOSS, filepath=saved_model, verbose=1, save_best_only=True)
checkpointer_log = ModelCheckpointWrapper(best_init=BEST_LOSS, filepath=logged_model, verbose=1, save_best_only=True, monitor='loss')
csvlogger = callbacks.CSVLogger(saved_log, separator=',', append=True)
history = model.fit_generator(
      train_generator,
      steps_per_epoch=_STEPS_PRE_EPOCH,
      epochs=_EPOCH_SIZE,
      validation_data=validation_generator,
      validation_steps= VALID_STEPS,
#       use_multiprocessing=True,
      callbacks = [checkpointer, tensorboard, lrscheduler, lrreducer, csvlogger, checkpointer_log],
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
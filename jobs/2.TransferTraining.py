from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input

from keras import models
from keras import layers
from keras import optimizers
from keras import callbacks
from datetime import datetime

IMAGE_SIZE = 299
CLASS_SIZE = 110
BATCH_SIZE = 8
VALIDATION_SPLIT = 0.05

TRAIN_SIZE = int(103350 * (1-VALIDATION_SPLIT))
EPOCH_SIZE = 10
STEPS_PRE_EPOCH = TRAIN_SIZE//BATCH_SIZE


#EPOCH_SIZE = 1
#STEPS_PRE_EPOCH = 10

print('STEPS_PRE_EPOCH {0}, BATCH_SIZE {1} '.format(STEPS_PRE_EPOCH, BATCH_SIZE))

train_dir = '../../official_data/IJCAI_2019_AAAC_train'
save_to_dir = None

# 載入權重
model_resnet50 = ResNet50(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

model = models.Sequential() # 產生一個新的網絡模型結構
model.add(model_resnet50)        # 把預訓練的卷積基底疊上去
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(CLASS_SIZE,activation='softmax',kernel_initializer='he_normal'))  # 疊上新的密集連接層來做為分類器

print('This is the number of trainable weights before freezing the conv base:', len(model.trainable_weights))
model_resnet50.trainable = False
print('This is the number of trainable weights after freezing the conv base:', len(model.trainable_weights))
model.summary()

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=VALIDATION_SPLIT)


train_generator = train_datagen.flow_from_directory(
        # 圖像資料的目錄
        train_dir,
        subset='training',
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        save_to_dir = save_to_dir,
        shuffle=True)

validation_generator = train_datagen.flow_from_directory(
        train_dir,
        subset='validation',
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical')

def lr_schedule(epoch):
    lr = 1e-4
    if epoch > 10:
        lr *= 1e-1
    elif epoch > 3:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr
model.compile(loss='categorical_crossentropy',
#               optimizer=optimizers.RMSprop(lr=2e-5),
              optimizer=optimizers.Adam(lr=2e-4),
              metrics=['accuracy'])


TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
train_log_dir = './logs/train/' + TIMESTAMP
test_log_dir = './logs/test/'   + TIMESTAMP

tensorboard = callbacks.TensorBoard(log_dir=train_log_dir, histogram_freq=10,
                          write_graph=True, write_images=False,  update_freq='batch')

history = model.fit_generator(
      train_generator,
      steps_per_epoch=STEPS_PRE_EPOCH,
      epochs=EPOCH_SIZE,
      validation_data=validation_generator,
      validation_steps=20,
#       use_multiprocessing=True,
      callbacks = [tensorboard],
      verbose=1)
      
model.save('trasfer_resnet50.h5') # 把模型儲存到檔案

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
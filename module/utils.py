from keras.preprocessing import image
import numpy as np
import os
import csv

def make_one_hot_array(data, n):
    return (np.arange(n)==data[:,None]).astype(np.integer)
def make_one_hot(data, n):
    return (np.arange(n)==data).astype(np.integer)
def calc_l2(x, xadv):
    diff = x.reshape((-1, 3)) - xadv.reshape((-1, 3))
    distance = np.mean(np.sqrt(np.sum((diff ** 2), axis=1)))
    return distance
def mem_data_generater(X,Y, batch_shape=None, label_shape=(None,110)):
    batch_size = batch_shape[0]
    input_size = batch_shape[2]
    label_size = label_shape[1]
    idx = 0
    while (True):
        s = idx*batch_size
        e = (idx+1)*batch_size
        if len(X[s:e]) == 0:
            break
        idx = idx+1
        yield None, X[s:e], Y[s:e]

def dev_data_generater(input_dir='../official_data/dev_data/', batch_shape=None, label_shape=(None,110)):
    batch_size = batch_shape[0]
    input_size = batch_shape[2]
    label_size = label_shape[1]
    images = np.zeros(batch_shape)
    labels = np.zeros((batch_size, label_size), dtype=np.int32)
    filenames = []
    idx = 0
    with open(os.path.join(input_dir, 'dev.csv')) as f:
        reader = csv.DictReader(f)
        for row in reader:
            filepath = os.path.join(input_dir, row['filename'])
            img = image.load_img(filepath, target_size=(input_size, input_size))
            img = image.img_to_array(img)
            images[idx] = img
            labels[idx] = make_one_hot(int(row['trueLabel']), label_size)
            filenames.append(os.path.basename(filepath))
            idx += 1
            if idx == batch_size:
                yield filenames, images, labels
                filenames = []
                images = np.zeros(batch_shape)
                labels = np.zeros((batch_size, label_size), dtype=np.int32)
                idx = 0
        if idx > 0:
            size = len(filenames)
            yield filenames, images[:size], labels[:size]


import time
class Profile():
    def __init__(self, output='', start=True):
        if start:
            self.start()
        self.output = output
    def start(self):
        self.begin_timestamp = time.time()
    def stop(self):
        self.end_timestamp = time.time()
        duration =  self.end_timestamp - self.begin_timestamp
        if self.output:
            print(self.output + " runs: %.2f s" % (duration))
        return duration


import matplotlib.pyplot as plt

def plot_images(X, Xadv, s=0, n=5):
    X_ = X.astype(np.uint8)
    Xadv_ = Xadv.astype(np.uint8)
    plt.figure(figsize=(10,10))
    for i in range(n):
        ii = 0+i
        plt.subplot(n,3, i*3+1)
        fig=plt.imshow(X_[ii])
        plt.subplot(n,3, i*3 + 2)
        fig=plt.imshow(Xadv_[ii])
        plt.subplot(n,3, i*3 + 3)
        diff = -np.abs(X_[ii].astype(np.int16) -  Xadv_[ii].astype(np.int16)) + 255
        fig=plt.imshow(diff.astype(np.uint8))
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
    plt.show()


from keras_preprocessing.image import array_to_img
class ImageSaver():
    def __init__(self, save_dir, data_format='channels_last', save_format='jpg', save_prefix='', scale=False):
        self.save_dir = save_dir
        self.data_format = data_format
        self.save_format = save_format
        self.save_prefix = save_prefix
        self.scale = scale
    def save_array(self, fname, image_array):
        if (fname[-4:] == '.jpg') or (fname[-4:] == '.png'):
            fname = fname[:-4]
        img = array_to_img(image_array, self.data_format, scale=self.scale)
        fname = '{fname}_{prefix}.{format}'.format(
            prefix=self.save_prefix,
            fname=fname,
            format=self.save_format)
        img.save(os.path.join(self.save_dir, fname))

def remove_with_name(path, pattern):
    count = 0
    for root, dirs, files in os.walk(path):
        for name in files: 
            pos = name.find(pattern)
            if (pos == -1):
                continue
            os.remove(os.path.join(root,name))
            count +=1
    return count

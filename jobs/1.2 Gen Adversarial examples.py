import sys
sys.path.append("..")
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from module.official_model_v3 import OfficialModel
OfficialModel.OFFICIAL_DATA_ROOT = '../../official_data/'

from module.EnhancedDataGenerator import *
from module.EmbeddedAttackModel import TargetModel, EmbeddedAttackModel
from module.gs_mim import GradSmoothMomentumIterativeMethod


from module.utils_tf import * 
from module.utils import * 
import tensorflow as tf
import time

IMAGE_SIZE = 299
BATCH_SIZE = 4
LABEL_SIZE = 110
good_dir = '../../official_data/prepared_train_data/good/'
adv_dir = '../../official_data/prepared_train_data/adv/'
batch_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3)


#create folder
import os
def might_create(save_dir, num_classes):
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)
        for i in range(num_classes):
            f = "0000"+str(i)
            os.makedirs(os.path.join(save_dir, f[-5:]))
might_create(adv_dir,LABEL_SIZE)

G = NamedDataGenerator()
GF = G.flow_from_directory(
    good_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size = BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    )

name = 'inception_v1'
T1 = TargetModel(batch_shape, LABEL_SIZE, name=name)
name = 'resnetv1_50'
T2 = TargetModel(batch_shape, LABEL_SIZE, name=name)
name = 'vgg_16'
T3 = TargetModel(batch_shape, LABEL_SIZE, name=name)

def generate(M, T, attack_params, max_sample_size = 5000, prefix='adv'):
    A = EmbeddedAttackModel(batch_shape, LABEL_SIZE)
    A.add_model(T)
    AdvSaver = ImageSaver(adv_dir, save_format='jpg', data_format=G.data_format, save_prefix=prefix)
    sample_size = 0
    check_timestamp = time.time()
    config = gpu_session_config()
    with tf.Session(config=config) as sess:
        A.attack_generate(sess, M, attack_params)
        for (X,Y),filenames in GF:
            Xadv = A.attack_batch(X, Y)
            for i in range(Xadv.shape[0]):
                sample_size+=1
                fname = filenames[i][:15]
                AdvSaver.save_array(fname, Xadv[i])
            if sample_size > max_sample_size:
                duration = time.time() - check_timestamp
                check_timestamp = time.time()
                print("%s total %d, duration %.2f mins" %(prefix, sample_size,duration/60))
                break
    tf.reset_default_graph()
    return X,Xadv
M = GradSmoothMomentumIterativeMethod


def generate_wrapper(M, T, epr,n, max_sample_size=1000):
    prefix = "{method}_{model}_ep{ep}_n{n}".format(method="gsmim", model=T.name, ep=int(epr*100), n=n) 
    attack_params = {"ep_ratio":epr, "nb_iter":n}
    X,Xadv = generate(M, T, attack_params, max_sample_size=max_sample_size, prefix=prefix)
    print(prefix + " last L2:", calc_l2(X,Xadv))
    return X,Xadv

gen_size = 1000
X,Xadv = generate_wrapper(M, T1, 0.02, 10, gen_size)
X,Xadv = generate_wrapper(M, T1, 0.06, 10, gen_size)
X,Xadv = generate_wrapper(M, T1, 0.1, 10, gen_size)

X,Xadv = generate_wrapper(M, T2, 0.02, 5, gen_size)
X,Xadv = generate_wrapper(M, T2, 0.06, 5, gen_size)
X,Xadv = generate_wrapper(M, T2, 0.1, 5, gen_size)

X,Xadv = generate_wrapper(M, T3, 0.02, 2, gen_size)
X,Xadv = generate_wrapper(M, T3, 0.06, 2, gen_size)
X,Xadv = generate_wrapper(M, T3, 0.1, 2, gen_size)
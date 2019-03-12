from keras.applications.nasnet import NASNetLarge, NASNetMobile
model = NASNetLarge(weights='imagenet', include_top=False, input_shape=(299,299,3))
# model = NASNetLarge(input_shape=(331,331,3), include_top=False, weights='imagenet')
model.summary()
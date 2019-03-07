
from keras.callbacks import TensorBoard, ModelCheckpoint
from datetime import datetime
import numpy as np

class TensorBoardWrapper(TensorBoard):
    '''Sets the self.validation_data property for use with TensorBoard callback.'''

    def __init__(self, batch_gen, nb_steps, b_size, **kwargs):
        super(TensorBoardWrapper, self).__init__(**kwargs)
        self.batch_gen = batch_gen # The generator.
        self.nb_steps = nb_steps   # Number of times to call next() on the generator.
        #self.batch_size = b_size

    def on_epoch_end(self, epoch, logs):
        # Fill in the `validation_data` property. Obviously this is specific to how your generator works.
        # Below is an example that yields images and classification tags.
        # After it's filled in, the regular on_epoch_end method has access to the validation_data.
        imgs, tags = None, None
        for s in range(self.nb_steps):
            ib, tb = next(self.batch_gen)
            if imgs is None and tags is None:
                imgs = np.zeros(((self.nb_steps * self.batch_size,) + ib.shape[1:]), dtype=np.float32)
                tags = np.zeros(((self.nb_steps * self.batch_size,) + tb.shape[1:]), dtype=np.uint8)
            imgs[s * ib.shape[0]:(s + 1) * ib.shape[0]] = ib
            tags[s * tb.shape[0]:(s + 1) * tb.shape[0]] = tb
        
        self.validation_data = [imgs, tags, np.ones(imgs.shape[0])]

                #debug
        tensors = (self.model.inputs +
                           self.model.targets +
                           self.model.sample_weights)
        if self.model.uses_learning_phase:
            self.validation_data += [np.ones(imgs.shape[0])]
        print("len", len(tensors), len(self.validation_data))
        # print("shape", self.model.inputs.shape, self.validation_data.shape)
              
        return super(TensorBoardWrapper, self).on_epoch_end(epoch, logs)

def TensorBoardCallback(gen, batch_size, log_dir):  
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    log_dir+= TIMESTAMP
    # tensorboard = TensorBoardWrapper(
                    # gen, 5, batch_size,
    tensorboard = TensorBoard(
                    log_dir=log_dir, histogram_freq=0,
                    write_graph=True, write_images=False,  update_freq='batch')
    return tensorboard


class ModelCheckpointWrapper(ModelCheckpoint):
    def __init__(self, best_init=None, *arg, **kwagrs):
        super().__init__(*arg, **kwagrs)
        if best_init is not None:
            self.best = best_init
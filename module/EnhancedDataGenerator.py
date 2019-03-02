from keras.preprocessing.image import ImageDataGenerator, DirectoryIterator
import numpy as np

class NamedDirectoryIterator(DirectoryIterator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filenames_np = np.array(self.filenames)
    def _get_batches_of_transformed_samples(self, index_array):
        return (super()._get_batches_of_transformed_samples(index_array),
            self.filenames_np[index_array])
class NamedDataGenerator(ImageDataGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def flow_from_directory(self,
                            directory,
                            target_size=(256, 256),
                            color_mode='rgb',
                            classes=None,
                            class_mode='categorical',
                            batch_size=32,
                            shuffle=True,
                            seed=None,
                            save_to_dir=None,
                            save_prefix='',
                            save_format='png',
                            follow_links=False,
                            subset=None,
                            interpolation='nearest'):
        return NamedDirectoryIterator(
            directory,
            self,
            target_size=target_size,
            color_mode=color_mode,
            classes=classes,
            class_mode=class_mode,
            data_format=self.data_format,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            follow_links=follow_links,
            subset=subset,
            interpolation=interpolation)

# DefenseDataGenerator

class DefenseDataGenerator(ImageDataGenerator):
    def __init__(self, msb_max=None, msb_rate=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.msb_max = msb_max
        self.msb_rate = msb_rate
        self.direc_flow = None
    def flow_from_directory(self, *args, **kwargs):
        self.direc_flow = super().flow_from_directory(*args, **kwargs)
        return self
    def __len__(self):
        return len(self.direc_flow)
    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)
    def next(self):
        X,Y = next(self.direc_flow)
        return self._msb_apply(X),Y
    def _msb_apply(self, x):
        if self.msb_max is not None:
            seed = np.random.randint(100)
            if seed < self.msb_rate*100:
                x = self.msb_defense(x)
        return x
    def _msb_random(self, _min=4, _max=32, n=None):
        return np.random.randint((_max-_min)//2+1, size=n)*2 + _min
    def msb_defense(self, x):
        msb = self._msb_random(4, self.msb_max, x.shape[0])
        msb = msb.reshape([x.shape[0],1,1,1])
        return (x//msb)*msb + (msb/2)


class MultiDataGenerator():
    # sources = {"good":{ "directory", "shuffle_num"}}
    def __init__(self, sources, *args, **kwargs):
        self.source_names = ['good', 'bad', 'adv']
        self.sources = sources
        if "msb_max" in kwargs:
            self.msb_max = kwargs["msb_max"]
            del kwargs["msb_max"]
        else:
            self.msb_max = None
        if "msb_rate" in kwargs:
            self.msb_rate = kwargs["msb_rate"]
            del kwargs["msb_rate"]
        else:
            self.msb_rate = None
        self._init_sources(sources, *args, **kwargs)
    def _init_sources(self, sources, *args, **kwargs):
        for s in self.source_names:
            if s in self.sources:
                self.sources[s]['generator'] = DefenseDataGenerator(msb_max=self.msb_max, msb_rate=self.msb_rate,*args, **kwargs)
    def get_train_flow(self, target_size, batch_size):
        shuffle_status = ""
        for s in self.source_names:
            if s in self.sources:
                directory = self.sources[s]['directory']
                self.sources[s]['train_flow'] = self.sources[s]['generator'].flow_from_directory(
                    directory, class_mode='categorical', shuffle=True, 
                    target_size=target_size, batch_size = batch_size, subset='training'
                )
                shuffle_rate = len(self.sources[s]['train_flow'])/self.sources[s]['shuffle_num']
                # print(len(self.sources[s]['train_flow']), self.sources[s]['shuffle_num'])
                shuffle_status +=  "%s: %.1f, "%(s, shuffle_rate)
        print("shuffle_status: ", shuffle_status)
        return self._train_flow()
    def _train_flow(self):
        while True:
            for s in self.source_names:
                if s in self.sources:
                    shuffle_num = self.sources[s]['shuffle_num']
                    for i in range(shuffle_num):
                        yield next(self.sources[s]['train_flow'])
    def get_valid_flow(self, target_size, batch_size):
        for s in self.source_names:
            if s in self.sources:
                directory = self.sources[s]['directory']
                self.sources[s]['valid_flow'] = self.sources[s]['generator'].flow_from_directory(
                    directory, class_mode='categorical', shuffle=True, 
                    target_size=target_size, batch_size = batch_size, subset='validation'
                )
        return self._valid_flow()
    def _valid_flow(self):
        while True:
            for s in self.source_names:
                if s in self.sources:
                    shuffle_num = self.sources[s]['shuffle_num']
                    for i in range(shuffle_num):
                        yield next(self.sources[s]['valid_flow'])

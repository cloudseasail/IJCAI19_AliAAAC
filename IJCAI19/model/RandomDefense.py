import numpy as np
from .ModelFactory import ModelFactory
from IJCAI19.module.EnhancedDataGenerator import DefenseDataGenerator

class DefenseModel():
    def __init__(self, output_size=None, name='', use_prob=True):
        self.name = name
        self.nb_classes = output_size
        self.use_prob = use_prob
        self.model = None
        if name:
            self.model = ModelFactory.create(name, output_size)
    def predict_create_graph(self, *arg, **kwargs):
        if self.model:
            return self.model.predict_create_graph(*arg, **kwargs)
    def evaluate_generator(self, *arg, **kwargs):
        if self.model:
            return self.model.evaluate_generator(*arg, **kwargs)
    def predict_batch(self, *arg, **kwargs):
        if self.model:
            return self.model.predict_batch(*arg, **kwargs)

class RandomDefense(DefenseModel):
    def __init__(self, output_size=None, name='', use_prob=True):
        super().__init__(output_size, name, use_prob)
        self.generator = None
    def random(self, *arg, **kwargs):
        self.generator = DefenseDataGenerator(*arg, **kwargs)
    def predict_create_graph(self, *arg, **kwargs):
        if self.generator is None:
            print("RandomDefense without random_transform")
        super().predict_create_graph(*arg, **kwargs)
    def _random_apply(self, x):
        if self.generator is None:
            return x
        ax = np.zeros_like(x)
        for i in range(x.shape[0]):
            ax[i] = self.generator.random_transform(x[i])
        return ax
    def predict_batch(self, X):
        x = self._random_apply(X)
        return super().predict_batch(x)
    def predict_generator(self, generator, batch_shape):
        x = self._random_apply(x)
        y = super().predict_generator(generator, batch_shape, use_prob=self.use_prob)
        return y
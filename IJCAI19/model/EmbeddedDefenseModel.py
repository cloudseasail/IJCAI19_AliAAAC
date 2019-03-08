import numpy as np
from .ModelFactory import ModelFactory
# from .RandomDefense import RandomDefense

class EmbeddedDefenseModel():
    def __init__(self, name=""):
        self.name = name
        self.models = []
        self.weights = []
    def add_model(self, model, weight=1.0):
        self.models.append(model)
        self.weights.append(weight)
    def predict_create_graph(self, batch_shape):
        for m in self.models:
            m.predict_create_graph(batch_shape, use_prob=True)
    def predict_batch(self, X, repeat=1.0):
        Yp_batch = []
        for idx in range(len(self.models)):
            m = self.models[idx]
            w = self.weights[idx]
            for ri in range(int(repeat)):
                yp = m.predict_batch(X)
                Yp_batch += ([yp*w])
        Yp_batch = np.mean(Yp_batch, axis=0)
        return Yp_batch
    def predict_generator(self, generator, batch_shape, repeat=1.0):
        Ypred = []
        Yprob = []
        self.predict_create_graph(batch_shape)
        for _,X,Y in generator:
            Yp_batch = self.predict_batch(X, repeat)
            Ypred+= [Yp_batch.argmax(1)]
            Yprob+= [Yp_batch.max(axis=1)]
        Ypred = np.concatenate(Ypred, axis=0)
        Yprob = np.concatenate(Yprob, axis=0)
        return Ypred,Yprob

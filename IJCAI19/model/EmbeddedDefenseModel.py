from .ModelFactory import ModelFactory

class EmbeddedDefenseModel():
    def __init__(self, name):
        self.name = name

class DefenseModel():
    def __init__(self, batch_shape=None, output_size=None, name='', use_prob=False):
        self.name = name
        self.model = None
        if name:
            self.model = ModelFactory.create(name, batch_shape, output_size, use_prob)

    def predict_preprocess(self, x):
        if self.model:
            return self.model.predict_preprocess(x)
    def attack_preprocess(self, x):
        if self.model:
            return self.model.attack_preprocess(x)
    def load_weight(self, *arg, **kwargs):
        if self.model:
            return self.model.load_weight(*arg, **kwargs)
    def get_endpoints(self, *arg, **kwargs):
        if self.model:
            return self.model.get_endpoints(*arg, **kwargs)

class MSBModel(DefenseModel):
    def __init__(self, msb=8, *arg, **kwargs):
        print("MSBModel", arg, kwargs)
        DefenseModel.__init__(self, *arg, **kwargs)
        self.msb = msb
    def predict_preprocess(self, x):
        x = self.msb_apply(x, self.msb)
        return super().predict_preprocess(x) 
    def msb_apply(self, x, msb):
        return (x//msb)*msb + (msb/2)

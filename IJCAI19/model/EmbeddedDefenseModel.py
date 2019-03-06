from .EmbeddedAttackModel import AttackModel
from .BatchModel import BatchModel
from .ModelFactory import ModelFactory

class EmbeddedDefenseModel():
    def __init__(self, name):
        self.name = name

class DefenseModel(BatchModel):
    def __init__(self, batch_shape=None, output_size=None, name='', use_prob=False):
        BatchModel.__init__(self, batch_shape=batch_shape, output_size=output_size, name=name, use_prob=use_prob)
        self.name = name
        self.model = None
        if name:
            self.model = ModelFactory(name)

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

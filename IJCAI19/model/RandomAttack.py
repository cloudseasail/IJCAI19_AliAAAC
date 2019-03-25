from IJCAI19.model.EmbeddedAttackModel import AttackModel
from IJCAI19.module.EnhancedDataGenerator import DefenseDataGenerator

class RandomAttack(AttackModel):
    def __init__(self, *arg, **kwargs):
        super().__init__( *arg, **kwargs)
        self.generator = None
    def random(self, *arg, **kwargs):
        self.generator = DefenseDataGenerator(*arg, **kwargs)
    def _random_apply(self, x):
        if self.generator is None:
            return x
        ax = np.zeros_like(x)
        for i in range(x.shape[0]):
            ax[i] = self.generator.random_transform(x[i])
        return ax
    def attack_preprocess(self, x):
        if self.generator is None:
            print("RandomAttack without random_transform")
        x = self._random_apply(x)
        return super().attack_preprocess(x)
import ConfigSpace

from src.search_space.base_component import BaseComponentSpace

# 0: [200], 1: [400], 2: [200, 100], 3: [400, 200], 4: [800, 400], 5: [200, 100, 50], 6: [400, 200, 100]
LAYERS_CHOICES = [0, 1, 2, 3, 4, 5, 6]


class FastAIMLPSpace(BaseComponentSpace):
    def __init__(self, component_hp, seed=0):
        super().__init__(component_hp, name="FastAIMLP", seed=seed)

        self.add_hyperparameters((
            ConfigSpace.CategoricalHyperparameter(
                f"{self.name}_layers",
                choices=LAYERS_CHOICES,
                default_value=0,
            ),
            ConfigSpace.UniformFloatHyperparameter(
                f"{self.name}_emb_drop",
                lower=0.0,
                upper=0.7,
                default_value=0.1,
            ),
            ConfigSpace.UniformFloatHyperparameter(
                f"{self.name}_ps",
                lower=0.0,
                upper=0.7,
                default_value=0.1,
            ),
            ConfigSpace.CategoricalHyperparameter(
                f"{self.name}_bs",
                choices=[256, 128, 512, 1024, 2048],
                default_value=256,
            ),
            ConfigSpace.UniformFloatHyperparameter(
                f"{self.name}_lr",
                lower=5e-4,
                upper=1e-1,
                default_value=1e-2,
                log=True,
            ),
            ConfigSpace.UniformIntegerHyperparameter(
                f"{self.name}_epochs",
                lower=20,
                upper=50,
                default_value=30,
            ),
        ))
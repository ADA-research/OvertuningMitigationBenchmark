import ConfigSpace
from src.search_space.base_component import BaseComponentSpace

class RealMLPSpace(BaseComponentSpace):
    """
    Defines the configuration space for the RealMLP model.

    This class builds the hyperparameter search space for the
    RealMLP_TD_Classifier via ConfigSpace.

    If relevant, default values are set on the value that has the highest probability in the paper's search space.
    We do not work with probabilities, since we will mostly do Bayesian optimization

    The search space can be found on page 45, table C14 of https://arxiv.org/pdf/2407.04491

    Inherits:
        BaseComponentSpace: Handles conditional addition of hyperparameters
        dependent on the model choice.
    """

    def __init__(self, component_hp, seed=0):
        """
        Initializes the RealMLP hyperparameter search space.

        Args:
            component_hp (ConfigSpace.hyperparameter.Hyperparameter):
                The top-level hyperparameter selector that identifies the model
                type (e.g., 'RealMLP').
        """
        super().__init__(component_hp, name="RealMLP", seed=seed)

        self.add_hyperparameters((

            # Architecture choice: index into predefined hidden size lists
            ConfigSpace.CategoricalHyperparameter(
                f"{self.name}_hidden_layer_sizes",
                choices=[0, 1, 2],  # 0: [256, 256, 256], 1: [64]*5, 2: [512]
                default_value=0
            ),

            # Numerical embedding type
            ConfigSpace.CategoricalHyperparameter(
                f"{self.name}_num_embedding_type",
                choices=[None, "pbld", "pl", "plr"],
                default_value=None
            ),

            # Activation function
            ConfigSpace.CategoricalHyperparameter(
                f"{self.name}_activation_function",
                choices=["relu", "selu", "mish"],
                default_value="relu"
            ),

            # Dropout probability
            ConfigSpace.CategoricalHyperparameter(
                f"{self.name}_dropout_prob",
                choices=[0.0, 0.15, 0.3],
                default_value=0.15
            ),

            # Weight decay
            ConfigSpace.CategoricalHyperparameter(
                f"{self.name}_weight_decay",
                choices=[0.0, 2e-2],
                default_value=0.0
            ),

            # Learning rate
            ConfigSpace.UniformFloatHyperparameter(
                f"{self.name}_learning_rate",
                lower=2e-2,
                upper=3e-1,
                log=True,
                default_value=5e-2
            ),

            # Optional front scaling layer
            ConfigSpace.CategoricalHyperparameter(
                f"{self.name}_use_scaling_layer",
                choices=[True, False],
                default_value=True
            ),

            # Weight initialization std (PLR sigma)
            ConfigSpace.UniformFloatHyperparameter(
                f"{self.name}_w_init_std",
                lower=0.05,
                upper=0.5,
                log=True,
                default_value=0.1
            ),
        ))

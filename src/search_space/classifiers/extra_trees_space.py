"""
Extra Trees Classifier Space Configuration.

This module defines the `ExtraTreesClassifierSpace` class, which is responsible
for constructing the hyperparameter search space for the ExtraTreesClassifier
model using ConfigSpace. All hyperparameters added to this space
are conditional on the model being set to 'ExtraTreesClassifier'.

Classes:
    - ExtraTreesClassifierSpace: Constructs the hyperparameter space for ExtraTreesClassifier.
"""

import ConfigSpace
from src.search_space.base_component import BaseComponentSpace

class ExtraTreesSpace(BaseComponentSpace):
    """
    Defines the configuration space for the ExtraTreesClassifier model.

    This class builds the search space for fitting an ExtraTreesClassifier model
    via ConfigSpace. It includes hyperparameters for controlling tree construction,
    regularization, and sampling behaviors.

    Inherits:
        BaseComponentSpace: Handles conditional addition of hyperparameters
        dependent on the model choice.

    Attributes:
        component_hp (ConfigSpace.hyperparameter.Hyperparameter):
            Reference to the top-level component hyperparameter, which determines
            the model type in the pipeline.
    """

    def __init__(self, component_hp, seed=0):
        """
        Initializes the ExtraTreesClassifier hyperparameter search space.

        Args:
            component_hp (ConfigSpace.hyperparameter.Hyperparameter):
                The top-level hyperparameter selector that identifies the model
                type (e.g., 'ExtraTreesClassifier').
        """
        super().__init__(component_hp, name="ExtraTrees", seed=seed)

        # TabArena ExtraTrees space
        self.add_hyperparameters((
            ConfigSpace.CategoricalHyperparameter(
                f"{self.name}_max_features", choices=["sqrt", 0.5, 0.75, 1.0], default_value="sqrt"
            ),
            ConfigSpace.UniformIntegerHyperparameter(
                f"{self.name}_min_samples_split", 2, 32, log=True
            ),
            ConfigSpace.Constant(f"{self.name}_bootstrap", False),
            ConfigSpace.Constant(f"{self.name}_n_estimators", 50),
            ConfigSpace.CategoricalHyperparameter(
                f"{self.name}_min_impurity_decrease", choices=[0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
            )
        ))

        # OLD SEARCH SPACE
        # # Add hyperparameters specific to ExtraTreesClassifier
        # self.add_hyperparameters((
        #     ConfigSpace.UniformIntegerHyperparameter(
        #         f"{self.name}_n_estimators", 10, 200, default_value=100
        #     ),
        #     ConfigSpace.CategoricalHyperparameter(
        #         f"{self.name}_criterion", choices=["gini", "entropy"], default_value="gini"
        #     ),
        #     ConfigSpace.CategoricalHyperparameter(
        #         f"{self.name}_max_features", choices=["sqrt", "log2"], default_value="sqrt"
        #     ),
        #     ConfigSpace.UniformIntegerHyperparameter(
        #         f"{self.name}_min_samples_split", 2, 20, default_value=2
        #     ),
        #     ConfigSpace.UniformIntegerHyperparameter(
        #         f"{self.name}_min_samples_leaf", 1, 20, default_value=1
        #     ),
        #     ConfigSpace.UniformFloatHyperparameter(
        #         f"{self.name}_min_weight_fraction_leaf", 0.0, 0.5, default_value=0.0
        #     ),
        #     ConfigSpace.UniformIntegerHyperparameter(
        #         f"{self.name}_max_leaf_nodes", 10, 1000, default_value=None
        #     ),
        #     ConfigSpace.UniformFloatHyperparameter(
        #         f"{self.name}_min_impurity_decrease", 0.0, 0.5, default_value=0.0
        #     ),
        #     ConfigSpace.CategoricalHyperparameter(
        #         f"{self.name}_bootstrap", choices=[True, False], default_value=False
        #     ),
        # ))
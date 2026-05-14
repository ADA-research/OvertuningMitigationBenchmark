"""
Linear Discriminant Analysis (LDA) Space Configuration.

This module defines the `LinearDiscriminantAnalysisSpace` class, which is responsible
for constructing the hyperparameter search space for the Linear Discriminant Analysis
(LDA) model using ConfigSpace. All hyperparameters added to this space
are conditional on the model being set to 'LinearDiscriminantAnalysis'.

Classes:
    - LinearDiscriminantAnalysisSpace: Constructs the hyperparameter space for LDA.
"""

import ConfigSpace
from src.search_space.base_component import BaseComponentSpace

class LinearDiscriminantAnalysisSpace(BaseComponentSpace):
    """
    Defines the configuration space for the Linear Discriminant Analysis (LDA) model.

    This class builds the search space for fitting an LDA model
    via ConfigSpace. It includes hyperparameters such as the solver
    and optional shrinkage for better performance in high-dimensional settings.

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
        Initializes the Linear Discriminant Analysis hyperparameter search space.

        Args:
            component_hp (ConfigSpace.hyperparameter.Hyperparameter):
                The top-level hyperparameter selector that identifies the model
                type (e.g., 'LinearDiscriminantAnalysis').
        """
        super().__init__(component_hp, name="LinearDiscriminantAnalysis", seed=seed)

        # Add hyperparameters specific to Linear Discriminant Analysis
        self.add_hyperparameters((
            ConfigSpace.CategoricalHyperparameter(
                f"{self.name}_solver", choices=["svd", "lsqr", "eigen"], default_value="svd"
            ),
            # ConfigSpace.CategoricalHyperparameter(
            #     f"{self.name}_shrinkage", [None, "auto"]
            # ),
        ))
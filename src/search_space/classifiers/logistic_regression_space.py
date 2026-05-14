"""
Logistic Regression Space Configuration.

This module defines the `LogisticRegressionSpace` class, which is responsible
for constructing the hyperparameter search space for the LogisticRegression
model using ConfigSpace. All hyperparameters added to this space
are conditional on the model being set to 'LogisticRegression'.

Classes:
    - LogisticRegressionSpace: Constructs the hyperparameter space for LogisticRegression.
"""

import ConfigSpace
from src.search_space.base_component import BaseComponentSpace

class LogisticRegressionSpace(BaseComponentSpace):
    """
    Defines the configuration space for the LogisticRegression model.

    This class builds the search space for fitting a LogisticRegression model
    via ConfigSpace. It includes hyperparameters such as the regularization
    strength and the solver type, which are constrained by the model being
    LogisticRegression in the specified pipeline.

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
        Initializes the LogisticRegression hyperparameter search space.

        Args:
            component_hp (ConfigSpace.hyperparameter.Hyperparameter):
                The top-level hyperparameter selector that identifies the model
                type (e.g., 'LogisticRegression').
        """
        super().__init__(component_hp, name="LogisticRegression", seed=seed)

        # Add hyperparameters specific to Logistic Regression
        self.add_hyperparameters((
            ConfigSpace.UniformFloatHyperparameter(
                f"{self.name}_C", 0.1, 1000, default_value=1.0, log=True
            ),
            ConfigSpace.CategoricalHyperparameter(
                f"{self.name}_solver",
                choices=["newton-cg", "liblinear", "lbfgs", "saga", "sag", "newton-cholesky"],
                default_value="lbfgs"
            )
        ))
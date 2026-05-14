import ConfigSpace


class BaseComponentSpace:
    def __init__(self, component_hp, name="base_component", seed=0):
        # Initialize base component search space
        self.space = ConfigSpace.ConfigurationSpace(name=name, seed=seed)

        # Add the pipeline component to the base space
        # Every hyperparameter added to this space is
        # conditional on the pipeline component being equal to the name of the search space
        self.space.add(component_hp)

        self.component_hp = component_hp
        self.name = name

    def add_hyperparameter(self, hp):
        """
        Adds a hyperparameter to the configuration space and includes a condition
        relating the newly added hyperparameter to the hyperparameter determining the component type choice.

        :param hp: A Hyperparameter object to be added to the configuration space.
        :type hp: ConfigSpace.hyperparameters.Hyperparameter
        """
        # Add a hyperparameter to the space
        self.space.add(hp)

        # Add condition for hyperparameter
        self.space.add(ConfigSpace.EqualsCondition(hp, self.component_hp, self.name))

    def add_hyperparameters(self, hps):
        """
        Adds multiple hyperparameters to the current instance by iterating through the provided
        list or collection and adding each hyperparameter individually using the
        `add_hyperparameter` method.

        :param hps: A collection of hyperparameters to be added.
        :type hps: list or iterable
        """
        for hp in hps:
            self.add_hyperparameter(hp)

    def get_hyperparameters(self):
        return [hp for hp in self.space.values() if hp.name != self.component_hp.name]

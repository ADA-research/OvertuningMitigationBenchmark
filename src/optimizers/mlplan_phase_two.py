from typing import List
import numpy as np

from src.optimizers import BaseOptimizer
from src.history.run_history import RunHistory
from src.history.data_classes import Run


class MLPlanPhaseTwoOptimizer(BaseOptimizer):
    """
    This optimizer simply performs the selection criteria out of the second phase of ML-Plan, and then one by one
    returns the configurations passing the selection criteria.

    Selection mechanism:
    k -> number of best and number of random configs to be included
    Random configs can deviate at most epsilon=0.03 from the optimal solution

    (Val) Score for configs in phase 2:
    0.5 * (search_score) + 0.5 * (select_score), where
    search score = average over 5MCCV
    select score = np.percentile(10MCCV scores, 75)
    """

    def __init__(
            self,
            run_history_phase_one: RunHistory,
            random_state: int = None,
            k: int = 25,
            epsilon: float = 0.03
    ):
        super().__init__(random_state=random_state)
        self.k = k
        self.epsilon = epsilon

        self.selected_configurations = self.select_configurations(run_history_phase_one)

        # Dict to store mapping from phase two to one
        self.iteration_map_phase_two_to_phase_one = {i: run.iteration for i, run in enumerate(self.selected_configurations)}

        # Initialize iteration counter
        self.iteration = -1

    def select_configurations(self, runs: RunHistory) -> List[Run]:
        """
        Select configurations from phase one history to be evaluated in phase two
        """
        # Select k top runs
        best_k_runs = sorted(runs.history, key=lambda l: l.average_val_score())[:self.k]

        # Filter runs that are in best k runs or are too much worse than incumbent
        configs_eligible_for_random_selection = [run for run in runs.history if
                                                 run not in best_k_runs and
                                                 run.average_val_score() <= runs.incumbent.average_val_score() + self.epsilon]

        # Select k random runs (or all runs if there are less than k random runs)
        if len(configs_eligible_for_random_selection) <= self.k:
            random_k_runs = configs_eligible_for_random_selection
        else:
            random_k_runs = list(np.random.choice(configs_eligible_for_random_selection, self.k, replace=False))

        # Return the runs to be included in phase 2
        return best_k_runs + random_k_runs

    def number_of_selected_configurations(self) -> int:
        """Function to get number of configs to evaluate in phase 2"""
        return len(self.selected_configurations)

    def phase_one_scores_of_selected_configurations(self) -> List[float]:
        """Function that returns a list of the selected configurations average validation scores."""
        return [run.average_val_score() for run in self.selected_configurations]

    def generate_configuration(self):
        # Increment counter en return the next configuration
        self.iteration += 1
        return self.selected_configurations[self.iteration].config, 0.0

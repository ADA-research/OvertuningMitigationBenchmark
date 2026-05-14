from src.mitigations.racing.base_racing import BaseRacing
import numpy as np
from src.history.data_classes import Run
from src.history.run_history import RunHistory

class EarlyStoppingCVBergman(BaseRacing):
    """
    Based on Paper and Repository (Follows the implementation of both) Attached is also the License

    https://github.com/automl/DontWasteYourTime-early-stopping/tree/main

    In the paper they mention that the EarlyStopping mechanism only checks before a configuration has been fully evaluated, however, according to the repository https://github.com/automl/amltk/blob/main/src/amltk/sklearn/evaluation.py#L874
    this is not the case.

    @inproceedings{bergman_dont_2024,
        title = {Don’t {Waste} {Your} {Time}: {Early} {Stopping} {Cross}-{Validation}},
        url = {https://proceedings.mlr.press/v256/bergman24a.html},
        booktitle = {Proceedings of the {Third} {International} {Conference} on {Automated} {Machine} {Learning}},
        publisher = {PMLR},
        author = {Bergman, Edward and Purucker, Lennart and Hutter, Frank},
        month = oct,
        year = {2024},
        pages = {9/1--31},
    }

    """
    def __init__(self,
                 aggressive: bool = False):
        """
        aggressive: True for Aggressive Racing, False for Forgiving Racing
        """
        super().__init__()
        self.aggressive = aggressive # Aggressive vs. Forgiving Racing

    def should_stop(self, candidate: Run, history: RunHistory = None, require_block: bool = True) -> bool:
        """
        Aggressive - Mean Fold Error of Incumbent
        Forgiving - Worst Fold Error of Incumbent (Highest for minimization objective, hence, max)
        """
        if history.incumbent is not None: # Alternatively, can use a patience period (in the repository referred to as "minimum_trials" however it was always set to 1 in the experiments)
            current_mean = candidate.average_val_score()
            if self.aggressive and  current_mean >= history.incumbent.average_val_score(): # For aggressive look at the average fold score
                candidate.early_stopped = True
                return True
            elif not self.aggressive and current_mean >= max([fold.scores.val for fold in history.incumbent.folds]): # For forgiving look at the worst score
                candidate.early_stopped = True
                return True
        return False
        

def top_n_stat(top_n: list = [], sigma: float = 1.0) -> float:
    """ Calculates the statistic used in the Robust Bergman Racing Strategy"""
    mean = np.mean(np.concatenate(top_n))
    std = sigma * np.std(np.concatenate(top_n)) # In Repo use ddof=0
    return mean + std

class RobustBergman(BaseRacing):
    """
    Based on Paper and Repository (Follows the implementation of both)

    The Robust Alternative to the Bergman Racing strategy in the paper they use either 3 or 5 configurations in the pool. The sigma term is used to control the aggressiveness of the stopping criterion.
    A higher sigma makes the Racing more aggressive, while a lower sigma makes it more forgiving. Note the sigma term is not mentioned in the paper, however, featured in the repository.

    In the paper they mention that the EarlyStopping mechanism only checks before a configuration has been fully evaluated, however, according to the repository https://github.com/automl/amltk/blob/main/src/amltk/sklearn/evaluation.py#L874
    this is not the case.

    https://github.com/automl/DontWasteYourTime-early-stopping/tree/main

    @inproceedings{bergman_dont_2024,
        title = {Don’t {Waste} {Your} {Time}: {Early} {Stopping} {Cross}-{Validation}},
        url = {https://proceedings.mlr.press/v256/bergman24a.html},
        booktitle = {Proceedings of the {Third} {International} {Conference} on {Automated} {Machine} {Learning}},
        publisher = {PMLR},
        author = {Bergman, Edward and Purucker, Lennart and Hutter, Frank},
        month = oct,
        year = {2024},
        pages = {9/1--31},
    }
    """
    def __init__(self,
                 n_configs: int = 5,
                 sigma: float = 1.0):
        """
        n_configs: size of the pool of configurations 3 is more aggressive, 5 is more forgiving (3 and 5 are the only two values checked)
        sigma: factor controls the aggressiveness of the stopping criterion (Default 1)
        """
        super().__init__()
        self.n_configs = n_configs
        self.sigma = sigma

    def should_stop(self, candidate: Run, history: RunHistory = None, require_block: bool = True) -> bool:

        # Need at least n_configs completed runs to start the race
        if len(history.history) < self.n_configs:
            return False

        # Initializing the first top_n configs
        if len(history.history) == self.n_configs:
            self.top_configs = []
            for i in range(self.n_configs):
                config = []
                for j in range(len(history.history[i].folds)):
                    config.append(history.history[i].folds[j].scores.val)
                self.top_configs.append(config)

        # Updating the top_n configurations after the early stopping decision of a candidate is made at the end of an evaluation
        if len(history.history) >= self.n_configs + 1:
            #Checking if the previous run should be added to the top_n only when on the last fold
            max_folds = len(history.incumbent.folds)
            current_best = self.top_configs
            current_best_score = top_n_stat(self.top_configs, self.sigma)
            cand = [[candidate.folds[j].scores.val for j in range(len(candidate.folds))]]
            # Try replacing each of the top_n configs with the candidate and see if it improves the statistic, if so continue evaluation, on last fold take the combination that gives the best improvement
            for i in range(self.n_configs):
                challenger = self.top_configs[:i] + cand + self.top_configs[i+1:]
                challenger_score = top_n_stat(challenger)
                if challenger_score < current_best_score:
                    if len(candidate.folds) < max_folds:
                        return False
                    current_best = challenger
                    current_best_score = challenger_score

            if current_best != self.top_configs and len(candidate.folds) == max_folds:
                self.top_configs = current_best
                return False
            candidate.early_stopped = True
            return True
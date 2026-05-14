from dataclasses import dataclass, field
from src.history.data_classes import Run
from src.history.run_history import RunHistory
from typing import List, Dict, Optional, Any

class BaseRacing:
    def __init__(self, required_configs: int = 1, elitist: bool = False, num_elites: int = 1):
        """
        required_configs: Number of Configurations per race iteration
        elitist: Binary switch for elitist racing for set based methods
        """
        self.required_configs = required_configs
        self.swap = False # Swapping incumbent and candidate in sequential setting
        self.terminate = False # Early stopping a Racing method
        self.elitist = elitist # Elitist Switch for set based methods
        self.num_elites = num_elites # Number of Elites to retain after each race iteration
    def should_stop(self, candidate: Run, history: RunHistory, require_block: bool = True) -> bool:
        pass
    def should_stop_set(self, candidates: Dict[int, Run], history: RunHistory, fold_ids: List[int], elites: List[int], evaluator: Any) -> Dict[int, Run]:
        pass
    def update_required_configs(self) -> None:
        pass





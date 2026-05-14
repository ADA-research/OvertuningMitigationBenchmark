from typing import List, Tuple


def calculate_trajectories(
        val_scores: List[float],
        test_scores: List[float],
        selection_scores: List[float] = None,
        scores_to_attach: dict = None
) -> Tuple[
    List[float],
    List[float],
    List[float],
    List[float],
    List[float],
    List[float],
    dict
]:
    # Iteration of current incumbent
    incumbent_iteration = 0
    best_test_incumbent_iteration = 0
    best_test_iteration = 0

    overtuning = []
    relative_overtuning = []
    meta_overfitting = []
    test_performances = []
    val_performances = []
    regret = []

    # Initialize scores_to_attach_dict, handling None values
    if scores_to_attach is None:
        scores_to_attach = {}

    scores_to_attach_dict = {
        k: [] for k, v in scores_to_attach.items() if v is not None
    }

    scores_to_select_incumbent_on = selection_scores if selection_scores is not None else val_scores

    for i in range(len(scores_to_select_incumbent_on)):
        if scores_to_select_incumbent_on[i] < scores_to_select_incumbent_on[incumbent_iteration]:
            incumbent_iteration = i

            if test_scores[i] < test_scores[best_test_incumbent_iteration]:
                best_test_incumbent_iteration = i

        # Update best test for regret calculation
        if test_scores[i] < test_scores[best_test_iteration]:
            best_test_iteration = i

        # Calculate positive overtuning. There is no negative overtuning (that is just a better configuration)
        current_overtuning = max(0.0, test_scores[incumbent_iteration] - test_scores[best_test_incumbent_iteration])
        overtuning.append(current_overtuning)

        # Relative overtuning = overtuning divided by the best achieved gain from the first configuration.
        # This follows the normalized overtuning idea and is well-defined only if the run achieved
        # positive improvement over the first incumbent at the current time step.
        max_gain_so_far = test_scores[0] - test_scores[best_test_incumbent_iteration]
        if max_gain_so_far > 0:
            relative_overtuning.append(current_overtuning / max_gain_so_far)
        elif current_overtuning > 0:
            relative_overtuning.append(float("inf"))
        else:
            relative_overtuning.append(0.0)

        # Calculate meta-overfitting; the difference between test and validation performance of the incumbent
        meta_overfitting.append(test_scores[incumbent_iteration] - val_scores[incumbent_iteration])

        # Calculate test regret; the difference between the test performance of the incumbent and the best test performance found so far
        regret.append(test_scores[incumbent_iteration] - test_scores[best_test_iteration])

        # Calculate incumbent performances
        test_performances.append(test_scores[incumbent_iteration])
        val_performances.append(val_scores[incumbent_iteration])

        # Attach incumbent information
        for k, v in scores_to_attach.items():
            if v is not None:
                scores_to_attach_dict[k].append(v[incumbent_iteration])

    return (
        val_performances,
        test_performances,
        meta_overfitting,
        overtuning,
        regret,
        relative_overtuning,
        scores_to_attach_dict,
    )





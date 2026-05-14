from ConfigSpace import ForbiddenEqualsClause, ForbiddenAndConjunction, ForbiddenGreaterThanRelation


def get_forbidden_clauses(config_space, problem_type):
    if problem_type == "binary":
        return get_binary_forbidden_clauses(config_space)
    elif problem_type == "multiclass":
        return get_multiclass_forbidden_clauses(config_space)
    else:
        return get_regression_forbidden_clauses(config_space)


def get_binary_forbidden_clauses(config_space):
    forbidden_clauses = []

    if config_space.get_hyperparameter('scaler').legal_value('Normalizer') and config_space.get_hyperparameter(
            'model').legal_value('MLP'):
        forbidden_clauses.append(ForbiddenAndConjunction(  # Normalizer not supported in combination with MLP
            ForbiddenEqualsClause(config_space.get_hyperparameter('scaler'), 'Normalizer'),
            ForbiddenEqualsClause(config_space.get_hyperparameter('model'), 'MLP')
        ))

    if config_space.get_hyperparameter('scaler').legal_value('None') and config_space.get_hyperparameter(
            'model').legal_value('MLP'):
        forbidden_clauses.append(
            ForbiddenAndConjunction(  # No scaler not supported in combination with MLP
                ForbiddenEqualsClause(config_space.get_hyperparameter('scaler'), 'None'),
                ForbiddenEqualsClause(config_space.get_hyperparameter('model'), 'MLP')
            ))

    if config_space.get_hyperparameter('dim_reducer').legal_value('FastICA') and config_space.get_hyperparameter(
            'scaler').legal_value('None'):
        forbidden_clauses.append(ForbiddenAndConjunction(  # No scaler not supported in combination with FastICA
            ForbiddenEqualsClause(config_space.get_hyperparameter('dim_reducer'), 'FastICA'),
            ForbiddenEqualsClause(config_space.get_hyperparameter('scaler'), 'None')
        ))

    # if config_space.get_hyperparameter('model').legal_value('LinearDiscriminantAnalysis'):
    #     forbidden_clauses.append(ForbiddenAndConjunction(
    #         ForbiddenEqualsClause(config_space.get_hyperparameter('model'), 'LinearDiscriminantAnalysis'),
    #         ForbiddenEqualsClause(config_space.get_hyperparameter('LinearDiscriminantAnalysis_solver'), 'svd'),
    #         ForbiddenEqualsClause(config_space.get_hyperparameter('LinearDiscriminantAnalysis_shrinkage'), 'auto')
    #     ))

    if config_space.get_hyperparameter('scaler').legal_value('RobustScaler'):
        forbidden_clauses.append(ForbiddenAndConjunction(
            ForbiddenEqualsClause(config_space.get_hyperparameter('scaler'), 'RobustScaler'),
            ForbiddenGreaterThanRelation(config_space.get_hyperparameter('RobustScaler_quantile_range_lower'),
                                         config_space.get_hyperparameter('RobustScaler_quantile_range_upper'))
        ))

    return forbidden_clauses


def get_multiclass_forbidden_clauses(config_space):
    forbidden_clauses = []

    if config_space.get_hyperparameter('scaler').legal_value('Normalizer') and config_space.get_hyperparameter(
            'model').legal_value('MLP'):
        forbidden_clauses.append(ForbiddenAndConjunction(  # Normalizer not supported in combination with MLP
            ForbiddenEqualsClause(config_space.get_hyperparameter('scaler'), 'Normalizer'),
            ForbiddenEqualsClause(config_space.get_hyperparameter('model'), 'MLP')
        ))

    if config_space.get_hyperparameter('scaler').legal_value('None') and config_space.get_hyperparameter(
            'model').legal_value('MLP'):
        forbidden_clauses.append(
            ForbiddenAndConjunction(  # No scaler not supported in combination with MLP
                ForbiddenEqualsClause(config_space.get_hyperparameter('scaler'), 'None'),
                ForbiddenEqualsClause(config_space.get_hyperparameter('model'), 'MLP')
            ))

    if config_space.get_hyperparameter('dim_reducer').legal_value('FastICA') and config_space.get_hyperparameter(
            'scaler').legal_value('None'):
        forbidden_clauses.append(ForbiddenAndConjunction(  # No scaler not supported in combination with FastICA
            ForbiddenEqualsClause(config_space.get_hyperparameter('dim_reducer'), 'FastICA'),
            ForbiddenEqualsClause(config_space.get_hyperparameter('scaler'), 'None')
        ))

    if config_space.get_hyperparameter('model').legal_value('Lasso'):
        forbidden_clauses.append(ForbiddenEqualsClause(config_space.get_hyperparameter("model"), "Lasso"))

    if config_space.get_hyperparameter('model').legal_value('ElasticNet'):
        forbidden_clauses.append(ForbiddenEqualsClause(config_space.get_hyperparameter("model"), "ElasticNet"))

    if config_space.get_hyperparameter('model').legal_value('GradientBoosting'):
        forbidden_clauses.append(ForbiddenAndConjunction(  # Exponential loss only for binary classification
            ForbiddenEqualsClause(config_space.get_hyperparameter('model'), 'GradientBoosting'),
            ForbiddenEqualsClause(config_space.get_hyperparameter('GradientBoosting_loss'), 'exponential')
        ))

    if config_space.get_hyperparameter('scaler').legal_value('RobustScaler'):
        forbidden_clauses.append(ForbiddenAndConjunction(
            ForbiddenEqualsClause(config_space.get_hyperparameter('scaler'), 'RobustScaler'),
            ForbiddenGreaterThanRelation(config_space.get_hyperparameter('RobustScaler_quantile_range_lower'),
                                         config_space.get_hyperparameter('RobustScaler_quantile_range_upper'))
        ))

    return forbidden_clauses


def get_regression_forbidden_clauses(config_space):
    forbidden_clauses = []

    if config_space.get_hyperparameter('scaler').legal_value('Normalizer') and config_space.get_hyperparameter(
            'model').legal_value('MLP'):
        forbidden_clauses.append(ForbiddenAndConjunction(  # Normalizer not supported in combination with MLP
            ForbiddenEqualsClause(config_space.get_hyperparameter('scaler'), 'Normalizer'),
            ForbiddenEqualsClause(config_space.get_hyperparameter('model'), 'MLP')
        ))

    if config_space.get_hyperparameter('scaler').legal_value('None') and config_space.get_hyperparameter(
            'model').legal_value('MLP'):
        forbidden_clauses.append(
            ForbiddenAndConjunction(  # No scaler not supported in combination with MLP
                ForbiddenEqualsClause(config_space.get_hyperparameter('scaler'), 'None'),
                ForbiddenEqualsClause(config_space.get_hyperparameter('model'), 'MLP')
            ))

    if config_space.get_hyperparameter('dim_reducer').legal_value('FastICA') and config_space.get_hyperparameter(
            'scaler').legal_value('None'):
        forbidden_clauses.append(ForbiddenAndConjunction(  # No scaler not supported in combination with FastICA
            ForbiddenEqualsClause(config_space.get_hyperparameter('dim_reducer'), 'FastICA'),
            ForbiddenEqualsClause(config_space.get_hyperparameter('scaler'), 'None')
        ))

    if config_space.get_hyperparameter('scaler').legal_value('RobustScaler'):
        forbidden_clauses.append(ForbiddenAndConjunction(
            ForbiddenEqualsClause(config_space.get_hyperparameter('scaler'), 'RobustScaler'),
            ForbiddenGreaterThanRelation(config_space.get_hyperparameter('RobustScaler_quantile_range_lower'),
                                         config_space.get_hyperparameter('RobustScaler_quantile_range_upper'))
        ))

    return forbidden_clauses

from sensai.evaluation import RegressionModelEvaluation, RegressionEvaluatorParams, VectorModelCrossValidatorParams
from sensai.util import logging
from laipred.data import Dataset
from laipred.model_factory import ModelFactory


def main():
    use_tsne = False
    test_registry = True
    use_cross_validation = True

    dataset = Dataset()
    io_data = dataset.load_io_data()

    models = None
    if use_tsne:
        models = [
            ModelFactory.create_pure_random_forest(),
            ModelFactory.create_pure_mlp(),
        ]
    elif test_registry:
        models = [
            ModelFactory.create_random_forest_collector(),
            ModelFactory.create_mlp_collector(),
        ]
    else:
        models = [
            ModelFactory.create_logistic_regression_orig(),
            ModelFactory.create_random_forest_orig(),
        ]

    # declare parameters to be used for evaluation, i.e. how to split the data (fraction and random seed)
    evaluator_params = RegressionEvaluatorParams(
        fractional_split_test_fraction=0.2, fractional_split_random_seed=42
    )
    cross_validator_params = VectorModelCrossValidatorParams(folds=3)
    # use a high-level utility class for evaluating the models based on these parameters
    ev = RegressionModelEvaluation(io_data, evaluator_params=evaluator_params,cross_validator_params=cross_validator_params)
    ev.compare_models(models,use_cross_validation=use_cross_validation)


if __name__ == "__main__":
    logging.run_main(main)

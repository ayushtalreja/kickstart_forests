from sklearn.preprocessing import StandardScaler

from .data import COL_WETNESS, COL_SENTINEL_VALUES
from sensai.data_transformation import DFTSkLearnTransformer, SkLearnTransformerFactoryFactory
from sensai.featuregen import FeatureGeneratorTakeColumns, FeatureCollector
from sensai.sklearn.sklearn_regression import (
    SkLearnMultiLayerPerceptronVectorRegressionModel,
    SkLearnRandomForestVectorRegressionModel,
    SkLearnLinearSVRVectorRegressionModel,
    SkLearnSVRVectorRegressionModel,
)
from .features import FeatureName, registry


class ModelFactory:
    COLS_USED_BY_ORIGINAL_MODELS = [COL_WETNESS, *COL_SENTINEL_VALUES]
    DEFAULT_FEATURES = (
        FeatureName.WETNESS,
        FeatureName.SENTINEL_LEVELS,
    )
    ALL_FEATURES = (FeatureName.WETNESS, FeatureName.SENTINEL_LEVELS,FeatureName.TREE_SPECIES)#, FeatureName.WAVELENGTH_LEVELS)

    @classmethod
    def create_logistic_regression_orig(cls):
        return (
            SkLearnLinearSVRVectorRegressionModel(solver="lbfgs", max_iter=1000)
            .with_feature_generator(
                FeatureGeneratorTakeColumns(cls.COLS_USED_BY_ORIGINAL_MODELS)
            )
            .with_name("LogisticRegression-orig")
        )

    @classmethod
    def create_random_forest_orig(cls):
        return (
            SkLearnRandomForestVectorRegressionModel(n_estimators=100)
            .with_feature_generator(
                FeatureGeneratorTakeColumns(cls.COLS_USED_BY_ORIGINAL_MODELS)
            )
            .with_name("RandomForest-orig")
        )

    @classmethod
    def create_pure_random_forest(cls):
        return (
            SkLearnRandomForestVectorRegressionModel(
                n_estimators=100, max_depth=100, random_state=42
            )
            .with_feature_generator(
                FeatureGeneratorTakeColumns(cls.COLS_USED_BY_ORIGINAL_MODELS)
            )
            .with_name("RandomForest-v2")
        )

    @classmethod
    def create_pure_mlp(cls):
        return (
            SkLearnMultiLayerPerceptronVectorRegressionModel(
                hidden_layer_sizes=(1000, 100, 10),
                max_iter=1000,
                random_state=42,
                early_stopping=True,
            )
            .with_feature_generator(
                FeatureGeneratorTakeColumns(cls.COLS_USED_BY_ORIGINAL_MODELS)
            )
            .with_name("MLP")
        )

    @classmethod
    def create_random_forest_collector(cls):
        fc = FeatureCollector(*cls.DEFAULT_FEATURES, registry=registry)
        return (
            SkLearnRandomForestVectorRegressionModel(
                n_estimators=100, max_depth=100, random_state=42
            )
            .with_feature_collector(fc)
            .with_name("RandomForest-collector")
        )

    @classmethod
    def create_mlp_collector(cls):
        fc = FeatureCollector(*cls.DEFAULT_FEATURES, registry=registry)
        return (
            SkLearnMultiLayerPerceptronVectorRegressionModel(
                hidden_layer_sizes=(1000, 100, 10),
                max_iter=1000,
                random_state=42,
                early_stopping=True,
            )
            .with_feature_collector(fc)
            # NOTE: For MLPs, normalisation/scaling for both features and targets should be considered
            #.with_feature_transformers(fc.create_feature_transformer_normalisation())
            #.with_target_transformer(DFTSkLearnTransformer(StandardScaler()))
            .with_name("MLP-collector")
        )

    @classmethod
    def create_cvr(cls):
        fc = FeatureCollector(*cls.ALL_FEATURES, registry=registry)
        return (SkLearnSVRVectorRegressionModel(
                C=100, epsilon=0.001, kernel='rbf',
            )
            .with_feature_collector(fc)
            .with_feature_transformers(fc.create_feature_transformer_normalisation(),
            fc.create_feature_transformer_one_hot_encoder())
            .with_target_transformer(DFTSkLearnTransformer(StandardScaler()))
            .with_name("SVR")
        )

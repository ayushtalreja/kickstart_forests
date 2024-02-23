from enum import Enum
from sklearn,manifold import TSNE

from .data import *
from sensai.data_transformation import (
    DFTNormalisation,
    SkLearnTransformerFactoryFactory,
)
from sensai.featuregen import (
    FeatureGeneratorRegistry,
    FeatureGeneratorTakeColumns,
    FeatureGenerator,
)


class FeatureName(Enum):
    WETNESS = "wetness"
    SENTINEL_LEVELS = "sentinel_levels"
    TREE_SPECIES = "tree_species"
    WAVELENGTH_LEVELS = "wavelength_levels"


class FeatureGeneratorTSNE(FeatureGenerator):
    def __init__(self):
        super().__init__(
            normalisation_rule_template=DFTNormalisation.RuleTemplate(
                transformer_factory=SkLearnTransformerFactoryFactory.MaxAbsScaler()
            )
        )
        self.col_feature = COL_WAVELENGTH_LEVELS
        self.col_sentinel = COL_SENTINEL_VALUES
        self.col_target = COL_LEAF_AREA_INDEX
        self.col_wetness = COL_WETNESS
        self._y = None

    def _fit(self, x: pd.DataFrame, y: pd.DataFrame = None, ctx=None):
        self._y = df[[self.col_target]]

        tsne_model = TSNE(n_components=3, learning_rate='auto', init='random', perplexity=30, random_state=0)
        wavelength_feature_data = x[self.col_feature]
        wavelength_embeddings = tsne_model.fit_transform(wavelength_feature_data)

        sentinel_feature_data = x.iloc[self.col_sentinel]
        tsne_model.set_params(n_components=3, perplexity=5)
        sentinel_embeddings = tsne_model.fit_transform(sentinel_feature_data)

        embeddings = np.concatenate((sentinel_embeddings, wavelength_embeddings), axis=1)

        wetness_features = x.iloc[self.col_wetness].values
        wetness_features = np.expand_dims(wetness_features, axis=1)
        embeddings = np.concatenate((embeddings, wetness_features), axis=1)
        self._values = pd.DataFrame(embeddings, axis=1)

    def _generate(self, df: pd.DataFrame, ctx=None) -> pd.DataFrame:
        # ctx: a context object whose functionality may be required for feature generation;
        #     this is typically the model instance that this feature generator is to generate inputs for
        ctx: SkLearnRandomForestVectorRegressionModel()
        is_training = ctx.is_being_fitted()
        
        tsne_model = TSNE(n_components=3, learning_rate='auto', init='random', perplexity=30, random_state=0)
        wavelength_feature_data = dataframe.iloc[self.col_feature]
        wavelength_embeddings = tsne_model.fit_transform(wavelength_feature_data)

        return pd.DataFrame({"wavelength_feature_data": wavelength_embeddings}, index=df.index)


registry = FeatureGeneratorRegistry()

registry.register_factory(
    FeatureName.WETNESS,
    lambda: FeatureGeneratorTakeColumns(
        COL_WETNESS,
        normalisation_rule_template=DFTNormalisation.RuleTemplate(skip=True),
    ),
)

registry.register_factory(
    FeatureName.SENTINEL_LEVELS,
    lambda: FeatureGeneratorTakeColumns(
        COL_SENTINEL_VALUES,
        normalisation_rule_template=DFTNormalisation.RuleTemplate(
            transformer_factory=SkLearnTransformerFactoryFactory.StandardScaler(),
            # NOTE: Normalisation of multiple features must state whether columns are to be handled independently
            #independent_columns=False,
        ),
    ),
)

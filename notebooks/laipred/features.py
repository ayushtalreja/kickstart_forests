from enum import Enum

from .data import *
from sensai.data_transformation import (
    DFTNormalisation,
    SkLearnTransformerFactoryFactory,
)
from sensai.featuregen import FeatureGeneratorRegistry, FeatureGeneratorTakeColumns


class FeatureName(Enum):
    WETNESS = "wetness"
    SENTINEL_LEVELS = "sentinel_levels"
    TREE_SPECIES = "tree_species"


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

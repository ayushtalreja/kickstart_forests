import logging
from typing import Optional
from sklearn.manifold import TSNE
from sensai import InputOutputData
from sensai.util.string import ToStringMixin
import pandas as pd
from . import config

log = logging.getLogger(__name__)

# Target Column
COL_LEAF_AREA_INDEX = "lai"

# Input Features
COL_WETNESS = "wetness"

COL_TREE_SPECIES = "treeSpecies"

COL_SENTINEL_2A_492 = "Sentinel_2A_492.4"
COL_SENTINEL_2A_559 = "Sentinel_2A_559.8"
COL_SENTINEL_2A_664 = "Sentinel_2A_664.6"
COL_SENTINEL_2A_704 = "Sentinel_2A_704.1"
COL_SENTINEL_2A_740 = "Sentinel_2A_740.5"
COL_SENTINEL_2A_782 = "Sentinel_2A_782.8"
COL_SENTINEL_2A_832 = "Sentinel_2A_832.8"
COL_SENTINEL_2A_864 = "Sentinel_2A_864.7"
COL_SENTINEL_2A_1613 = "Sentinel_2A_1613.7"
COL_SENTINEL_2A_2202 = "Sentinel_2A_2202.4"

COL_SENTINEL_VALUES = [
    COL_SENTINEL_2A_492,
    COL_SENTINEL_2A_559,
    COL_SENTINEL_2A_664,
    COL_SENTINEL_2A_704,
    COL_SENTINEL_2A_740,
    COL_SENTINEL_2A_782,
    COL_SENTINEL_2A_832,
    COL_SENTINEL_2A_864,
    COL_SENTINEL_2A_1613,
    COL_SENTINEL_2A_2202,
]


class Dataset(ToStringMixin):
    def __init__(
        self,
        num_samples: Optional[int] = None,
        drop_na: bool = True,
        drop_tree_species: bool = True,
        random_seed: int = 42,
    ):
        """
        :param num_samples: the number of samples to draw from the data frame; if None, use all samples
        :param drop_na: whether to drop null values or not
        :param drop_tree_species: whether to drop tree species or not
        :param random_seed: the random seed to use when sampling data points
        """
        self.num_samples = num_samples
        self.drop_na = drop_na
        self.drop_tree_species = drop_tree_species
        self.random_seed = random_seed

    def load_data_frame(self) -> pd.DataFrame:
        """
        :return: the full data frame for this dataset (including the class column)
        """

        csv_path = config.csv_data_path()
        log.info(f"Loading {self} from {csv_path}")

        if self.drop_na:
            df = pd.read_csv(csv_path, index_col=0).dropna()
        else:
            df = pd.read_csv(csv_path, index_col=0)

        if self.drop_tree_species:
            df.drop("treeSpecies", axis=1, inplace=True)

        if self.num_samples is not None:
            df = df.sample(self.num_samples, random_state=self.random_seed)

        column_names = list(df.columns)
        self.wavelength_columns = [x for x in column_names if x.startswith("w")][1:]
        return df

    def load_io_data(self) -> InputOutputData:
        """
        :return: the I/O data
        """
        return InputOutputData.from_data_frame(
            self.load_data_frame(), COL_LEAF_AREA_INDEX
        )

    # Guowen remove your tsne from here and add it into model.py. This was part of step 3.

    # def load_xy(self, use_tsne: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
    #     """
    #     :return: a pair (X, y) where X is the data frame containing all attributes and y is the corresponding series of class values
    #     """
    #     df = self.load_data_frame()

    #     #
    #     if use_tsne:
    #         tsne_model = TSNE(
    #             n_components=3,
    #             learning_rate="auto",
    #             init="random",
    #             perplexity=30,
    #             random_state=0,
    #         )
    #         wavelength_feature_data = df.loc[:, self.wavelength_columns]
    #         wavelength_embeddings = tsne_model.fit_transform(wavelength_feature_data)

    #         sentinel_feature_data = df.loc[:, COL_SENTINEL_VALUES]
    #         tsne_model.set_params(n_components=3, perplexity=5)
    #         sentinel_embeddings = tsne_model.fit_transform(sentinel_feature_data)

    #         embeddings = np.concatenate(
    #             (sentinel_embeddings, wavelength_embeddings), axis=1
    #         )

    #         wetness_features = df.loc[:, COL_WETNESS].values
    #         wetness_features = np.expand_dims(wetness_features, axis=1)
    #         embeddings = np.concatenate((embeddings, wetness_features), axis=1)
    #         return embeddings, df[COL_LEAF_AREA_INDEX].values
    #     else:
    #         return df.drop(columns=COL_LEAF_AREA_INDEX), df[COL_LEAF_AREA_INDEX]

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

COL_WAVELENGTH_LEVELS = []
for i in range(400, 2501):
    COL_WAVELENGTH_LEVELS.append("w" + str(i))


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

        df = pd.read_csv(csv_path, index_col=0).dropna()
        if self.drop_na:
            df = df.dropna()

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

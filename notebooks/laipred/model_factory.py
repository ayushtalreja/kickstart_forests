from sklearn import linear_model
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor


from.data import COL_WETNESS, COL_SENTINEL_VALUES

class ModelFactory:
    COLS_USED_BY_ORIGINAL_MODELS = [COL_WETNESS, *COL_SENTINEL_VALUES]

    @classmethod
    def create_logistic_regression_orig(cls):
        return Pipeline(
            [
                (
                    "project_scale",
                    ColumnTransformer(
                        [("scaler", StandardScaler(), cls.COLS_USED_BY_ORIGINAL_MODELS)]
                    ),
                ),
                ("model", linear_model.LinearRegression()),
            ]
        )

    @classmethod
    def create_random_forest_orig(cls):
        return Pipeline(
            [
                (
                    "project_scale",
                    ColumnTransformer(
                        [("scaler", StandardScaler(), cls.COLS_USED_BY_ORIGINAL_MODELS)]
                    ),
                ),
                ("model", RandomForestRegressor(n_estimators=100)),
            ]
        )

    @classmethod
    def pure_random_forest(cls):
        return RandomForestRegressor(n_estimators=100, max_depth=100, random_state=42)

    @classmethod
    def pure_mlp(cls):
        return MLPRegressor(
            hidden_layer_sizes=(1000, 100, 10),
            max_iter=1000,
            random_state=42,
            early_stopping=True,
        )

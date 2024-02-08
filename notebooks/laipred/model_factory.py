from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor


from.data import COL_WETNESS, COL_SENTINEL_VALUES

class ModelFactory:
    COLS_USED_BY_ORIGINAL_MODELS = [COL_WETNESS, *COL_SENTINEL_VALUES]

    @classmethod
    def create_linear_regression_orig(cls):
        return Pipeline(
            [
                ("model", linear_model.LinearRegression()),
            ]
        )

    @classmethod
    def create_random_forest_orig(cls):
        return Pipeline(
            [
                ("model", RandomForestRegressor(n_estimators=100)),
            ]
        )

    @classmethod
    def pure_random_forest(cls):
                return Pipeline(
            [
                ("model", RandomForestRegressor(n_estimators=100, max_depth=100, random_state=42)),
            ]
        )


    @classmethod
    def pure_mlp(cls):
        return Pipeline(
            [
                ("model",MLPRegressor(
            hidden_layer_sizes=(1000, 100, 10),
            max_iter=1000,
            random_state=42,
            early_stopping=True,)),
            ]
        )



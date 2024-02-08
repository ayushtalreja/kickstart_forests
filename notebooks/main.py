from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

from laipred.data import Dataset
from laipred.model_factory import ModelFactory

def main():
    use_tsne = True

    dataset = Dataset()
    X, y = dataset.load_xy(use_tsne=use_tsne)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    models = (
        [
            ModelFactory.create_linear_regression_orig(),
            ModelFactory.create_random_forest_orig(),
        ]
        if not use_tsne
        else [ModelFactory.pure_random_forest(), ModelFactory.pure_mlp()]
    )

    # evaluate models
    for model in models:
        print(f"Evaluating model:\n{model}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"R2_score: {r2_score(y_test, y_pred)}")
        print(f"MSE: {mean_squared_error(y_test, y_pred)}")
if __name__ == "__main__":
    main()


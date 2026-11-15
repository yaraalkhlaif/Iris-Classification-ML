import pandas as pd
from sklearn.datasets import load_iris


class DataLoader:

    def __init__(self):
        self.data = None
        self.df = None

    def load_data(self):
        iris = load_iris()

        self.df = pd.DataFrame(
            iris.data,
            columns=iris.feature_names
        )

        self.df["target"] = iris.target
        self.data = iris

        return self.df

    def explore_data(self):
        print("First 5 rows:")
        print(self.df.head())

        print("\nDataset Info:")
        print(self.df.info())

        print("\nStatistical Summary:")
        print(self.df.describe())

        print("\nMissing Values:")
        print(self.df.isnull().sum())

    def split_features_labels(self):
        X = self.df.drop("target", axis=1)
        y = self.df["target"]

        return X, y

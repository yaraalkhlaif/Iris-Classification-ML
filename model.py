from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


class ModelTrainer:

    def __init__(self):
        self.model = LogisticRegression(max_iter=200)

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def split_data(self, X, y):

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_model(self):

        self.model.fit(self.X_train, self.y_train)
        return self.model


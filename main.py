from data_loader import DataLoader
from visualization import Visualization
from model import ModelTrainer
from sklearn.metrics import accuracy_score

# Load and explore data
loader = DataLoader()
df = loader.load_data()
loader.explore_data()

X, y = loader.split_features_labels()

# Visualization
viz = Visualization(df)
viz.scatter_plot()
viz.histogram()

# Split and train model
trainer = ModelTrainer()

X_train, X_test, y_train, y_test = trainer.split_data(X, y)

model = trainer.train_model()

print("\nModel trained successfully!")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Predict on test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

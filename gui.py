import tkinter as tk
import pandas as pd
from tkinter import messagebox
from data_loader import DataLoader
from model import ModelTrainer

# Load and train model
loader = DataLoader()

df = loader.load_data()

X, y = loader.split_features_labels()

trainer = ModelTrainer()

trainer.split_data(X, y)

model = trainer.train_model()

# GUI Window
window = tk.Tk()
window.title("Iris Flower Predictor")
window.geometry("400x400")

# Labels and Entry Boxes
tk.Label(window, text="Sepal Length").pack()
entry1 = tk.Entry(window)
entry1.pack()

tk.Label(window, text="Sepal Width").pack()
entry2 = tk.Entry(window)
entry2.pack()

tk.Label(window, text="Petal Length").pack()
entry3 = tk.Entry(window)
entry3.pack()

tk.Label(window, text="Petal Width").pack()
entry4 = tk.Entry(window)
entry4.pack()

# Prediction Function
def predict_flower():

    try:

        values = pd.DataFrame([[
            float(entry1.get()),
            float(entry2.get()),
            float(entry3.get()),
            float(entry4.get())
        ]], columns=X.columns)

        prediction = model.predict(values)

        flower_names = [
            "Setosa",
            "Versicolor",
            "Virginica"
        ]

        result = flower_names[prediction[0]]
        messagebox.showinfo(
            "Prediction",
            f"The flower is: {result}"
        )

    except:

        messagebox.showerror(
            "Error",
            "Please enter valid numbers"
        )


# Predict Button
tk.Button(
    window,
    text="Predict",
    command=predict_flower
).pack(pady=20)

window.mainloop()

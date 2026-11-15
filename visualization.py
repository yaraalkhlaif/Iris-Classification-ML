import matplotlib.pyplot as plt

class Visualization:

    def __init__(self, df):

        self.df = df


    def scatter_plot(self):

        plt.figure()

        plt.scatter(

            self.df["petal length (cm)"],

            self.df["petal width (cm)"],

            c=self.df["target"]

        )

        plt.title("Petal Length vs Petal Width")

        plt.xlabel("Petal Length (cm)")

        plt.ylabel("Petal Width (cm)")

        plt.show()



    def histogram(self):

        plt.figure()

        plt.hist(self.df["sepal length (cm)"], bins=20)

        plt.title("Sepal Length Distribution")

        plt.xlabel("Sepal Length (cm)")

        plt.ylabel("Frequency")

        plt.show()


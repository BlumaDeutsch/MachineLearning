import seaborn as sns
import matplotlib.pyplot as plt


def show_heatmap(data):
    plt.figure(figsize=(15, 10))
    sns.heatmap(data.corr(), annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.show()

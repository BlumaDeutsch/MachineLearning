from Functions import run_model, processing_data
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    data = pd.read_csv("UCI_Credit_Card.csv")
    print("First 5 rows:")
    print(data.head())
    print(data.columns)

    processing_data(data)



if __name__ == '__main__':
    main()


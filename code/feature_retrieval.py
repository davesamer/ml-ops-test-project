import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    df = pd.read_csv("data/churn_data_raw.csv")
    y = df.churn
    df = df.drop('churn', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(df, y, train_size=0.7, random_state=123)

    X_train.to_csv("data/X_train.csv")
    X_test.to_csv("data/X_test.csv")
    y_train.to_csv("data/y_train.csv")
    y_test.to_csv("data/y_test.csv")


    
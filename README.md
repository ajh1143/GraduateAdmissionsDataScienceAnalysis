# Kaggle Dataset Analysis - Graduate Admissions
*In Progress*

## Imports
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    import matplotlib.pyplot as plt

## Set file path
    file_path = '...Admission_Predict.csv'

## Create DataFrame from source file
    df = pd.read_csv(file_path)

## EDA
    df.info()
    print(df.describe())

## Extract features, target
    target = df["Chance of Admit "].values
    features = df.drop(["Chance of Admit ", 'Serial No.'],axis=1)

## Viz
    features.hist()
    plt.show()

## test, train
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.30, random_state = 1)

## instantiate LR
    lr = LinearRegression()

## fit data
    lr.fit(X_train, y_train)

## Predict
    y_pred = lr.predict(X_test)

## Performance Metrics
    print(lr.score(X_test, y_test))

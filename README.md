# Kaggle Dataset Analysis - Graduate Admissions
*In Progress*

## Dataset Columns
Serial No.	    
GRE Score    
TOEFL Score    
University Rating    
SOP	    
LOR	    
CGPA    	
Research    	
Chance of Admit

________________________________________________________________________________________________________________________________________
## Imports
```Python3
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    import matplotlib.pyplot as plt
```

________________________________________________________________________________________________________________________________________
## Set file path
```Python3
    file_path = '...Admission_Predict.csv'
```

________________________________________________________________________________________________________________________________________
## Create DataFrame from source file
```Python3
    df = pd.read_csv(file_path)
```

________________________________________________________________________________________________________________________________________
## Exploratory Data Analysis
Generate columns, data types, counts    
Generate summary statistics
```Python3
    df.info()
    print(df.describe())
```

________________________________________________________________________________________________________________________________________
## Extract features, target
Target: Probability of admission    
Features: Drop target, drop `Serial No.` (used as index)
```Python3
    target = df["Chance of Admit "].values
    features = df.drop(["Chance of Admit ", 'Serial No.'],axis=1)
```

________________________________________________________________________________________________________________________________________
## Viz
Plot Histogram distributions of features, with axes labeled for each plot
```Python3
for each in features:
    features[each].hist()
    plt.title(each+ ' distribution')
    plt.xlabel(each)
    plt.ylabel("Observations\n(Count)")
    plt.show()
```

________________________________________________________________________________________________________________________________________
## test, train
```Python3
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.30, random_state = 1)
```

________________________________________________________________________________________________________________________________________
## Instantiate LinearRegression()
```Python3
    lr = LinearRegression()
```

________________________________________________________________________________________________________________________________________
## Fit Data
```Python3
    lr.fit(X_train, y_train)
```

________________________________________________________________________________________________________________________________________
## Predict
```Python3
    y_pred = lr.predict(X_test)
```

________________________________________________________________________________________________________________________________________
## Performance Metrics
### Accuracy Score
```Python3
    print(lr.score(X_test, y_test))
```

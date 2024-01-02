import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv('bmi.csv')
data.head()
data = data.drop(['Index'], axis=1)
data['BMI'] = data['Weight'] / (data['Height'] / 100) ** 2
data['BMI'] = data['BMI'].round(2)

X = data[['Height', 'Weight']]
y = data['BMI']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

pickle.dump(model, open("model.pkl", "wb"))

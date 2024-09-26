from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import  pandas as pd

file = pd.read_csv('Iris.csv')
file.head(5) 

x = file[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = file['Species'] 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .3, random_state=42)

model = LogisticRegression(random_state=42)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)   

print(f'Accuracy: {accuracy:.3f}')

print(f'\n classification Report:')

print(classification_report(y_test, y_pred, target_names=file['Species'].unique()))

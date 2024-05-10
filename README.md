# experiment-13

Ex.No: 13 Machine Learning – Mini Project

DATE: 

REGISTER NUMBER : 

AIM:
To write a program to train the classifier for Diabetes.

Algorithm:
Step 1: Import packages. Step 2: Get the data. Step 3: Split the data. Step 4: Scale the data. Step 5: Instantiate model. Step 6: Create Gradio Function. Step 7: Print Result.

Program:
import numpy as np
import pandas as pd
pip install gradio
pip install typing-extensions --upgrade
pip install --upgrade typing
pip install typing-extensions --upgrade
import gradio as gr
data = pd.read_csv('/content/diabetes.csv')
data.head()
print(data.columns)
x = data.drop(['Outcome'], axis=1)
y = data['Outcome']
print(x[:5])
#split data
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test= train_test_split(x,y)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)

from sklearn.neural_network import MLPClassifier
model = MLPClassifier(max_iter=1000, alpha=1)
model.fit(x_train, y_train)
print("Model Accuracy on training set:", model.score(x_train, y_train))
print("Model Accuracy on Test Set:", model.score(x_test, y_test))

def diabetes(Pregnancies, Glucose, Blood_Pressure, SkinThickness, Insulin, BMI,Diabetes_Pedigree, Age):
    x = np.array([Pregnancies,Glucose,Blood_Pressure,SkinThickness,Insulin,BMI,Diabetes_Pedigree,Age])
    prediction = model.predict(x.reshape(1, -1))
    if(prediction==0):
      return "NO"
    else:
      return "YES"

outputs = gr.Textbox()
app = gr.Interface(fn=diabetes, inputs=['number','number','number','number','number','number','number','number'], outputs=outputs,description="Detection of Diabeties")
app.launch(share=True)

Output:

![image](https://github.com/niranjanninja/experiment-13/assets/148364568/ec0bb56e-b9e6-4bba-a520-686eeb0e4f57)

![image](https://github.com/niranjanninja/experiment-13/assets/148364568/0ad3d662-4309-4c6e-a371-495a18d0a2a5)

![image](https://github.com/niranjanninja/experiment-13/assets/148364568/0e4d7e74-70b8-4997-96a3-2181d25e264f)




Result:
Thus the system was trained successfully and the prediction was carried out

import uvicorn
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
df = pd.read_csv('creditcard.csv')
# create an instance of KNeighborsClassifier
# model = KNeighborsClassifier()
x = df.drop('Class', axis=1)
y = df['Class']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
# fit the model with training data
# knn.fit(X_train, y_train)

# make predictions on test data
# y_pred = knn.predict(X_test)

# import pandas
# import pandas
# import sklearn

# print(pandas._version_)
# print(sklearn._version_)
# Load the pre-trained model
with open('trained_knn.pkl', 'rb') as f:
    model = pickle.load(f)


# try:
#     with open('train3.pkl', 'rb') as f:
#         data = pickle.load(f)
# except (pickle.UnpicklingError, EOFError, ValueError) as e:
#     print(f"The pickle file is corrupt: {e}")
# else:
#     print("The pickle file is not corrupt")

# with open('train3.pkl', 'rb') as f:
#     X_train, y_train = pickle.load(f)

# Fit the model with training data
# model.fit(X_train, y_train)
# Define the FastAPI app
app = FastAPI()

# Define the HTML form
html_form = """
<body>
  <form action="/predict" method="post">
    V1: <input type="text" name="V1"><br>
    V2: <input type="text" name="V2"><br>
    V3: <input type="text" name="V3"><br>
    V4: <input type="text" name="V4"><br>
    V5: <input type="text" name="V5"><br>
    V6: <input type="text" name="V6"><br>
    V7: <input type="text" name="V7"><br>
    V8: <input type="text" name="V8"><br>
    V9: <input type="text" name="V9"><br>
    V10: <input type="text" name="V10"><br>
    V11: <input type="text" name="V11"><br>
    V12: <input type="text" name="V12"><br>
    V13: <input type="text" name="V13"><br>
    V14: <input type="text" name="V14"><br>
    V15: <input type="text" name="V15"><br>
    V16: <input type="text" name="V16"><br>
    V17: <input type="text" name="V17"><br>
    V18: <input type="text" name="V18"><br>
    V19: <input type="text" name="V19"><br>
    V20: <input type="text" name="V20"><br>
    V21: <input type="text" name="V21"><br>
    V22: <input type="text" name="V22"><br>
    V23: <input type="text" name="V23"><br>
    V24: <input type="text" name="V24"><br>
    V25: <input type="text" name="V25"><br>
    V26: <input type="text" name="V26"><br>
    V27: <input type="text" name="V27"><br>
    V28: <input type="text" name="V28"><br>
     V29: <input type="text" name="V29"><br>
      V30: <input type="text" name="V30"><br>
    <input type="submit" value="Submit">
  </form>
</body>
"""

# Define the API endpoint
@app.get("/")
async def root():
    return HTMLResponse(content=html_form, status_code=200)

@app.post('/predict')
async def predict_fraud(V1: float = Form(...),
                        V2: float = Form(...),
                        V3: float = Form(...),
                        V4: float = Form(...),
                        V5: float = Form(...),
                        V6: float = Form(...),
                        V7: float = Form(...),
                        V8: float = Form(...),
                        V9: float = Form(...),
                        V10: float = Form(...),
                        V11: float = Form(...),
                        V12: float = Form(...),
                        V13: float = Form(...),
                        V14: float = Form(...),
                        V15: float = Form(...),
                        V16: float = Form(...),
                        V17: float = Form(...),
                        V18: float = Form(...),
                        V19: float = Form(...),
                        V20: float = Form(...),
                        V21: float = Form(...),
                        V22: float = Form(...),
                        V23: float = Form(...),
                        V24: float = Form(...),
                        V25: float = Form(...),
                        V26: float = Form(...),
                        V27: float = Form(...),
                        V28: float = Form(...),
                        V29:float=   Form(...),
                        V30:float=   Form(...)):
    
    # Create a numpy array with the input data
    X = np.array([V1, V2, V3, V4, V5, V6, V7, V8, V9, V10,
                  V11, V12, V13, V14, V15, V16, V17, V18, V19, V20,
                  V21, V22, V23, V24, V25, V26, V27, V28,V29,V30]).reshape(1, -1)

    # Predict the probability of fraud

    

    
    y_proba = knn.predict_proba(X)[0, 1]
    
    # Define the threshold to classify a transaction as fraudulent
    threshold = 0.5
    
    # Check if the transaction is fraudulent
    if y_proba > threshold:
        result = 'fraudulent'
    else:
        result = 'not fraudulent'
        
    # Return the prediction result
    return {'prediction': result, 'probability': y_proba}
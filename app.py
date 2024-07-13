from fastapi import FastAPI
from StudentInput import StudentInfo
import numpy as np
import pandas as pd
import pickle as pk
from sklearn.preprocessing import StandardScaler
import os
from utils import clip, create_feature, scaling, label_encoding, cols_to_remove,class_mapping
app = FastAPI()

# Load the pretrained model
model_path = "graduate_classifier.pkl"
scaler_path="scaler.pkl"


if os.path.exists(model_path):
    print(f"File size: {os.path.getsize(model_path)} bytes")
    with open(model_path, "rb") as pkl_in:
        model = pk.load(pkl_in)
else:
    raise FileNotFoundError("Model file does not exist")


# loading scaler.pkl

if os.path.exists(scaler_path):
    print(f"File size: {os.path.getsize(scaler_path)}")
    with open(scaler_path,"rb") as sc_pkl:
        test_scaler=pk.load(sc_pkl)
else:
    raise FileNotFoundError("File doesn't exis")



@app.post("/predict")
def predict(student: StudentInfo):
    input_data = student.dict(by_alias=True)
    input_data = pd.DataFrame([input_data])
    
    # Preprocessing
    input_data = clip(input_data)
    input_data = create_feature(input_data)
    input_data = label_encoding(input_data, train=False)
    cols = cols_to_remove()
    input_data = input_data.drop(columns=cols, axis=1)
    input_data = scaling(test_df=input_data, is_train=False)
    
    # Predictions
    predictions = model.predict(input_data)
    predicted_class = np.argmax(predictions, axis=1)
    predicted_class = class_mapping(predicted_class)
    
    return {"prediction": predicted_class}



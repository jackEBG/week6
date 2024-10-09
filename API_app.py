from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the trained model
model = joblib.load('model.joblib')

# Create FastAPI instance
app = FastAPI()

# Define the root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the ML Prediction API!"}

# Define a favicon endpoint
@app.get("/favicon.ico")
def favicon():
    return {"message": "No favicon available."}

# Define the input data structure
class PredictionRequest(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    # Add other features as per your model

# Define the prediction endpoint
@app.post('/predict')
def predict(request: PredictionRequest):
    input_data = np.array([[request.feature1, request.feature2, request.feature3]])
    prediction = model.predict(input_data)
    return {'prediction': prediction[0]}

# Run the application
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
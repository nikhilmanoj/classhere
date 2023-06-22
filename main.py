import pickle
from fastapi import FastAPI
from pydantic import BaseModel



# Define the FastAPI app
app = FastAPI()

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define the input data schema
class TestScores(BaseModel):
    test_score1: int
    test_score2: int
    test_score3: int
    test_score4: int

# Define the prediction route
@app.post("/predict")
def predict_learning_style(test_scores: TestScores):
    scores = [test_scores.test_score1, test_scores.test_score2, test_scores.test_score3, test_scores.test_score4]
    prediction = model.predict([scores])
    return {"predicted_learning_style": prediction[0]}




@app.get("/")
async def root():
    return {"message": "Hello World"}




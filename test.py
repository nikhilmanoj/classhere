import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle

# Load the dataset
data = pd.read_csv('dataset.csv')

# Prepare the data
X = data.iloc[:, :-1]  # Features (test scores)
y = data.iloc[:, -1]   # Target variable (learning style)

# Train the model
model = DecisionTreeClassifier()
model.fit(X, y)

# Get input from the user for test scores
test_scores = []
for i in range(4):
    score = int(input(f"Enter test score {i+1}: "))
    test_scores.append(score)

# Make prediction
prediction = model.predict([test_scores])
print("Predicted learning style:", prediction[0])

# Save the trained model
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

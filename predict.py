import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import sys

# Define the neural network model
class XORNet(nn.Module):
    def __init__(self):
        super(XORNet, self).__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 2)
        self.fc3 = nn.Linear(2, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Load the model
def load_model(path="two_input_xor_nn.pth"):
    model = XORNet()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

# Function to preprocess input data
def preprocess_data(X, scaler):
    return scaler.transform(X)

def predict_xor(model, input_data, scaler):
    # Preprocess the input data
    X = preprocess_data(np.array(input_data).reshape(1, -1), scaler)
    
    # Convert to tensor
    X = torch.tensor(X, dtype=torch.float32)
    
    # Predict the output using the model
    with torch.no_grad():
        prediction = model(X)
    
    # Return the predicted value
    return prediction.item()

def calculate_accuracy(model, scaler):
    # Define the XOR inputs and expected outputs
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    expected_outputs = np.array([0, 1, 1, 0])
    
    # Preprocess the inputs
    inputs_scaled = preprocess_data(inputs, scaler)
    
    # Predict the outputs
    inputs_tensor = torch.tensor(inputs_scaled, dtype=torch.float32)
    with torch.no_grad():
        predictions = model(inputs_tensor).numpy().flatten()
    
    # Convert predictions to binary outputs
    predictions = (predictions > 0.5).astype(int)
    
    # Calculate accuracy
    accuracy = np.mean(predictions == expected_outputs)
    return accuracy, predictions

if __name__ == "__main__":
    # Load the scaler and the model
    _, _, scaler = load_and_preprocess_data()
    model = load_model()

    # Calculate accuracy
    accuracy, predictions = calculate_accuracy(model, scaler)
    
    # Print predictions and accuracy
    print("Predictions for XOR inputs [0, 0], [0, 1], [1, 0], [1, 1]:")
    print(predictions)
    print(f"Model accuracy: {accuracy * 100:.2f}%")

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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

# Load and preprocess data
def load_and_preprocess_data():
    df = pd.read_csv("train.csv")
    X = df.iloc[:, :2].values
    y = df.iloc[:, 2].values.reshape(-1, 1)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y, scaler

# Train the model
def train_model(model, criterion, optimizer, X_train, y_train, num_epochs=5000, tol=1e-20):
    for epoch in range(num_epochs):
        model.train()
        
        inputs = torch.tensor(X_train, dtype=torch.float32)
        labels = torch.tensor(y_train, dtype=torch.float32)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}')
        
        if loss.item() < tol:
            break

# Save the model
def save_model(model, path="two_input_xor_nn.pth"):
    torch.save(model.state_dict(), path)

if __name__ == "__main__":
    X, y, scaler = load_and_preprocess_data()
    
    model = XORNet()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    
    train_model(model, criterion, optimizer, X, y)
    save_model(model)

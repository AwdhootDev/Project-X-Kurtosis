import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim

print("Loading dataset...")
loaded_data = np.load("cwru_dataset.npz")
X_data = loaded_data['X']
y_data = loaded_data['y']

X_train, X_test, y_train, y_test = train_test_split(
    X_data, 
    y_data, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_data
)

# print("X_train shape:", X_train.shape)
# print("X_test shape:", X_test.shape)
# print("y_train shape:", y_train.shape)
# print("y_test shape:", y_test.shape)

X_train_3d = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_3d = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]) )

X_train_tensor = torch.tensor(X_train_3d, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_3d, dtype = torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset( X_test_tensor, y_test_tensor)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

data_iterator = iter(train_loader)
batch_X, batch_y = next(data_iterator)
# print("Batch X shape:", batch_X.shape)
# print("Batch y shape:", batch_y.shape)

class FaultDetectionCNN(nn.Module):
    def __init__(self):
        super(FaultDetectionCNN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Flatten(),
            nn.Linear(in_features=3840, out_features=4)
        )

    def forward(self, x):
        return self.network(x)

model = FaultDetectionCNN()
test_output = model(batch_X)
print("Output shape:", test_output.shape)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
print("Starting Training...")

for epoch in range(num_epochs):
    
    running_loss = 0.0
    
    for inputs, labels in train_loader:
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

print("Training Complete!")

print("\nStarting Evaluation on Test Set...")

model.eval()

correct_predictions = 0
total_predictions = 0

with torch.no_grad():

    for inputs, labels in test_loader:

        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

accuracy = (correct_predictions / total_predictions) * 100
print(f"Final Test Accuracy: {accuracy:.2f}%")
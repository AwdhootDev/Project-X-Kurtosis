import numpy as np 
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

train_data = np.load("cwru_train_multi_load.npz")
test_data = np.load("cwru_test_3HP.npz")

X_train, y_train = train_data['X'], train_data['y']
X_test, y_test = test_data['X'], test_data['y']

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

class BearingDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]

train_dataset = BearingDataset(X_train_tensor, y_train_tensor)
test_dataset = BearingDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class FaultDetectionCNN(nn.Module):
    def __init__(self):
        super(FaultDetectionCNN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=16),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.3),

            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=8),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.3),

            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=4),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.25),

            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=4),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=16, out_channels=8, kernel_size=4),
            nn.Tanh(),
            nn.Dropout(0.25)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(960, 128),
            nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.network(x)
        x = self.classifier(x)
        return x

model = FaultDetectionCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

num_epochs = 30
print("Starting Training...")

for epoch in range(num_epochs):

    model.train()
    
    running_loss = 0.0
    total = 0
    correct = 0
    
    for inputs, labels in train_loader:
        
        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    print(f"Accuracy : {epoch_accuracy:.2f}%")

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
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lifelines.utils import concordance_index

# Load the dataset
df = pd.read_excel("..\radiomicsC2.xlsx")

# Rename the last two columns
df.columns = [*df.columns[:-2], 'survival_time', 'survival_status']

# Extract features and survival information
X = df.iloc[:, 1:-2]
y = df[['survival_time', 'survival_status']]  # Survival time and status

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data preprocessing
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_time_train_tensor = torch.tensor(y_train['survival_time'].values, dtype=torch.float32)
y_event_train_tensor = torch.tensor(y_train['survival_status'].values, dtype=torch.float32)

T = 120  # This value should be set according to the number of time bins in your dataset

# Define the DeepHit model
class DeepHitNet(nn.Module):
    def __init__(self, input_dim, num_time_bins):
        super(DeepHitNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.dropout1 = nn.Dropout(p=0.7)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(p=0.7)
        self.fc3 = nn.Linear(256, 128)
        self.out = nn.Linear(128, num_time_bins)  # Output layer has T units

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        return torch.softmax(self.out(x), dim=1)  # Use softmax to get the probability distribution

# Instantiate the model
input_dim = X_train.shape[1]
model = DeepHitNet(input_dim, T)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define loss function and optimizer
def deephit_loss(pred_probs, y_time, y_event, alpha=1, beta=1, sigma=10):
    # Compute the log-likelihood loss
    y_time_one_hot = torch.nn.functional.one_hot(y_time.long(), num_classes=T)
    log_likelihood = -torch.sum(y_event * torch.log(torch.sum(pred_probs * y_time_one_hot, dim=1) + 1e-8))

    # Compute the ranking loss
    risk = torch.sum(pred_probs * torch.arange(1, T + 1).view(1, -1), dim=1)  # Predicted risk scores
    event_mask = y_event.view(-1, 1)
    risk_diff = risk.view(-1, 1) - risk.view(1, -1)
    time_diff = y_time.view(-1, 1) < y_time.view(1, -1)
    event_time_mask = event_mask * time_diff
    ranking_loss = -torch.mean(torch.exp(-risk_diff / sigma) * event_time_mask)

    # Compute the calibration loss
    calibration_loss = torch.sum((pred_probs - y_time_one_hot) ** 2)

    # Total loss
    total_loss = alpha * log_likelihood + beta * ranking_loss + calibration_loss

    return total_loss

optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Train the model
num_epochs = 2000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    pred_probs = model(X_train_tensor)
    loss = deephit_loss(pred_probs, y_time_train_tensor, y_event_train_tensor)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')
    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            pred_probs_test = model(X_test_tensor)
            # Compute the expected survival time for each sample
            expected_time = torch.sum(pred_probs_test * torch.arange(1, T + 1).view(1, -1), dim=1)
            ci = concordance_index(y_test['survival_time'], expected_time.numpy(), y_test['survival_status'])
            print(f'Concordance Index on test set: {ci}')

# Evaluate the model
model.eval()
with torch.no_grad():
    pred_probs_test = model(X_test_tensor)
    # Compute the expected survival time for each sample
    expected_time = torch.sum(pred_probs_test * torch.arange(1, T + 1).view(1, -1), dim=1)
    ci = concordance_index(y_test['survival_time'], expected_time.numpy(), y_test['survival_status'])
    print(f'Concordance Index on test set: {ci}')

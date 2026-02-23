import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


        
abnormal_df = pd.read_csv("ptbdb_abnormal.csv", header=None)
normal_df = pd.read_csv("ptbdb_normal.csv", header=None)

df = pd.concat([abnormal_df, normal_df], ignore_index=True) 
df = df.sample(frac=1, random_state=42).reset_index(drop=True) 

x = df.iloc[:, :-1].to_numpy(dtype=np.float32)
y = df.iloc[:, -1].to_numpy(dtype=np.int64)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2, stratify=y) 
input_scaler = StandardScaler()


x_train_scaled = input_scaler.fit_transform(x_train)
x_test_scaled = input_scaler.transform(x_test)
joblib.dump(input_scaler, "ABNscaler.joblib")


xtr = torch.tensor(x_train_scaled, dtype=torch.float32)
ytr = torch.tensor(y_train, dtype=torch.long) 
xte = torch.tensor(x_test_scaled, dtype=torch.float32)
yte = torch.tensor(y_test, dtype=torch.long)


class Model(nn.Module):
    def __init__(self, in_features=187, h1=10, h2=8, out_features=2):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1) 
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features) 


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x
torch.manual_seed(42)
model = Model(in_features=xtr.shape[1], out_features=2) 
criterion = nn.CrossEntropyLoss()

opt = torch.optim.Adam(model.parameters(), lr=0.01) 



epochs = 100
losses = []
model.train() 
for i in range(epochs):
    opt.zero_grad()
    y_pred = model(xtr)
    loss = criterion(y_pred, ytr) 
    loss.backward()
    opt.step()
    losses.append(loss.item())
    if i % 10 == 0: 
        print(f"Epoch: {i}, Loss: {loss.item(): .4f}")

model.eval()    
with torch.no_grad():
    logits = model(xte)
    pred = logits.argmax(dim=1)
    correct = (pred==yte).float() 
    test_acc = correct.mean().item() 
    test_acc_percent = test_acc * 100
    print(f'Validation Accuracy: {test_acc_percent:.4f}%')

sns.set_theme(style="darkgrid")
sns.lineplot(x=range(len(losses)), y=losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()

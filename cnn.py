import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import torch
from torch.utils.data import Dataset, DataLoader


class Model(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, conv_stride, pool_kernel, pool_stride, num_layers, h1, h2, output_layer):
        super(Model, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(in_features, out_features, kernel_size, conv_stride, padding)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(out_features, out_features, kernel_size, conv_stride, padding)
        self.layer1 = nn.Sequential(
            self.conv1,
            self.relu1,
            self.conv2
        )
        self.num_layers = num_layers
        self.maxpool1 = nn.MaxPool1d(pool_kernel, pool_stride)
        self.conv3 = nn.Conv1d(out_features, out_features, kernel_size, conv_stride, padding)
        self.relu2 = nn.ReLU()
        self.layer2 = nn.Sequential(
            self.conv3,
            self.relu2
        )
        self.maxpool2 = nn.MaxPool1d(pool_kernel, pool_stride)
        self.conv4 = nn.Conv1d(out_features, out_features, kernel_size, conv_stride, padding)
        self.relu3 = nn.ReLU()
        self.flatten = nn.Flatten(1)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(out_features, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, output_layer)
        )


    def forward(self, x):
        fwd = self.layer1(x)
        fwd = self.maxpool1(fwd)
        for _ in range(self.num_layers):
            fwd = self.layer2(fwd)
        fwd = self.maxpool2(fwd)
        fwd = self.conv4(fwd)
        fwd = self.relu3(fwd)
        fwd = self.gap(fwd)
        fwd = self.flatten(fwd)

        fwd = self.fc(fwd)

        return fwd


class ECGDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


train_df = pd.read_csv("mitbih_train.csv")
test_df = pd.read_csv("mitbih_test.csv")
train_df = train_df.dropna()
test_df = test_df.dropna()



features_train = train_df.iloc[:, :-1].to_numpy(dtype=np.float32)
labels_train = train_df.iloc[:, -1].to_numpy(dtype=np.int64)

features_test = test_df.iloc[:, :-1].to_numpy(dtype=np.float32)
labels_test = test_df.iloc[:, -1].to_numpy(dtype=np.int64)

input_scaler = StandardScaler()


x_train = input_scaler.fit_transform(features_train)
x_test = input_scaler.transform(features_test)

joblib.dump(input_scaler, "ARRscaler")


xtr = torch.tensor(x_train, dtype=torch.float32)
xte = torch.tensor(x_test, dtype=torch.float32)
ytr = torch.tensor(labels_train, dtype=torch.long)
yte = torch.tensor(labels_test, dtype=torch.long)

xtr = xtr.unsqueeze(1)
xte = xte.unsqueeze(1)


train_dataset = ECGDataset(xtr, ytr)
test_dataset = ECGDataset(xte, yte)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model = Model(in_features=1, out_features=32, kernel_size=15, conv_stride=1, pool_kernel=2, pool_stride=2, num_layers=3, h1=30, h2=10, output_layer=5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def train_one_epoch():
    model.train(True)
    print(f"Epoch: {epoch +1}")
    running_loss = 0.0
    running_loss_100 = []
    loss_e = []
    for batch_idx, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        output = model(x_batch)
        loss = loss_function(output, y_batch)
        running_loss += loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_e.append(loss.item())
        if batch_idx % 100 == 99:
            avg_loss = running_loss / 100
            print(f'Batch{batch_idx+1} Loss: {avg_loss}')
            running_loss_100.append(avg_loss)
            running_loss = 0.0
    print()
    return  loss_e, running_loss_100

def validate_one_epoch():
    model.train(False)
    running_loss = 0.0
    total_correct = 0
    total_samples = 0
    for batch_idx, batch in enumerate(test_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        with torch.no_grad():
            output = model(x_batch)
            pred = torch.argmax(output, axis=1)
            total_correct += (pred == y_batch).sum().item()
            total_samples += y_batch.size(0)
            loss = loss_function(output, y_batch) 
            running_loss += loss.item()
    avg_loss = running_loss / len(test_loader)
    test_acc = (total_correct / total_samples) * 100
    print(f"Val Loss: {avg_loss:.3f} Accuracy: {test_acc: 0.3f}")
    print("************************************")
    print()
    return avg_loss, test_acc


num_epochs = 10

loss_function = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.001)
val = []
loss_plot = []
loss_100_plot = []
acc = []
for epoch in range(num_epochs):
    loss_e, running_loss_100 = train_one_epoch()
    val_loss, test_acc = validate_one_epoch()
    val.append(val_loss)
    acc.append(test_acc)
loss_plot.extend(loss_e)
loss_100_plot.extend(running_loss_100)





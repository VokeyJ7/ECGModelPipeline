import torch
import torch.nn as nn


class Model(nn.Module):
    def __init___(in_features, out_features, kernel_size, conv_stride, padding, pool_stride, num_layers, h1, h2, softmax_dim, output_layer):
        super(Model, self).__init__()
        
        self.conv1 = nn.Conv1d(in_features, out_features, kernel_size, conv_stride, padding)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(out_features, out_features, kernel_size, conv_stride, padding)
        self.layer1 = nn.Sequential(
            self.conv1(),
            self.relu1(),
            self.conv2()
        )
        
        self.maxpool1 = nn.MaxPool1d(out_features, pool_stride)
        self.conv3 = nn.Conv1d(out_features, out_features, kernel_size, conv_stride, padding)
        self.relu2 = nn.ReLU()
        self.layer2 = nn.Sequential(
            self.conv3(),
            self.relu2()
        )
        self.maxpool2 = nn.MaxPool1d(out_features, pool_stride)
        self.conv3(out_features, out_features, kernel_size, conv_stride, padding),
        self.relu2()
        
        self.conv4 = nn.Conv1d(out_features, out_features, kernel_size, conv_stride, padding)
        self.relu3 = nn.ReLU()
        self.fc = nn.Sequential(
            nn.Linear(out_features, h1),
            nn.Linear(h1, h2),
            nn.Linear(h2, output_layer)
        )
        self.softmax = nn.Softmax(dim=softmax_dim)

    def forward(self, x):
        fwd = self.layer1(x)
        fwd = self.maxpool1(fwd)
        for _ in num_layers:
            fwd = self.layer2(fwd)
        fwd = self.maxpool2(fwd)   
        fwd = self.conv4(fwd)
        fwd = self.relu3(fwd)
        fwd = self.fc(fwd)
        out = self.softmax(fwd)
        

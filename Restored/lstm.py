import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Attention(nn.Module):
    def __init__(self, hidden_d, input_d) -> None:
        super(Attention, self).__init__()
        self.layer1 = nn.Linear(hidden_d, 1)
        self.layer2 = nn.Linear(input_d, 1)

        self.V = nn.Parameter(torch.rand(hidden_d, 1), requires_grad=True)

    def forward(self, c, x):
        score =F.tanh(self.layer1(c)) + self.layer2(x)
        attention_weights = F.softmax(self.V * score, dim=0)

        context_vector = attention_weights * c
        context_vector = torch.sum(context_vector, dim=0)
        return context_vector, attention_weights

class Network(nn.Module):
    def __init__(self, input_d, hidden_d, layer_d, output_d):
        super(Network, self).__init__()
        self.hidden_dim = hidden_d
        self.layer_dim = layer_d
        self.lstm = nn.LSTM(input_d + hidden_d, hidden_d, layer_d, batch_first=True).to(device)
        self.fc = nn.Linear(hidden_d, output_d)
        self.attention = Attention(hidden_d, input_d)

    def forward(self, img, x, h0 = None, c0 = None):
        if h0 == None:
          h0 = torch.zeros(self.hidden_dim, device=device, requires_grad=True)
          h0 = h0.unsqueeze(0)
        if c0 == None:
          c0 = torch.zeros(self.hidden_dim, device=device, requires_grad=True)
          c0 = c0.unsqueeze(0)
        # Adding Attention to the image embedding and the previous hidden state
        img_1, h0 = self.attention(img, h0)
        x = torch.cat((x, img_1), dim=0)

        x = x.unsqueeze(0)
        # print("x: ", x.shape)
        # print("h: ", h0.shape)
        # print("c: ", c0.shape)

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        out1 = self.fc(out).to(device)
        return out1, hn,  cn

    def configure_optimizers(self):
     return Adam(self.parameters(), lr = 0.01)
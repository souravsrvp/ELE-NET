import torch
import torch.nn as nn

class ELE_Net(nn.Module):
    def __init__(self, input_dim, embed_dim, output_classes):
        super(ELE_Net, self).__init__()
        self.dropout = nn.Dropout(p=0.2)
        self.LSTM = nn.LSTM(input_size=input_dim, hidden_size=embed_dim, num_layers=4, batch_first=True)
        self.Linear1 = nn.Linear(embed_dim, embed_dim//2)
        self.act = nn.PReLU()
        self.Linear2a = nn.Linear(embed_dim//2, output_classes)
        self.Linear2b = nn.Linear(embed_dim//2, output_classes)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        x = self.dropout(input)
        x, _ = self.LSTM(x)
        x = self.dropout(x)
        x = x[:, -1, :]
        x = self.Linear1(x)
        x = self.act(x)
        x1 = self.Linear2a(x)
        x2 = self.Linear2b(x)
        #x = self.softmax(x)
        return x1, x2
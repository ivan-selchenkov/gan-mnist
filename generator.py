import torch
from torch import nn

criterion = nn.BCEWithLogitsLoss()


class Generator(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size):
        super(Generator, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, hidden_dim * 4)

        self.fc4 = nn.Linear(hidden_dim * 4, output_size)

        self.dropout = nn.Dropout(p=0.3)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.tahn = nn.Tanh()

    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))
        x = self.dropout(x)

        x = self.leaky_relu(self.fc2(x))
        x = self.dropout(x)

        x = self.leaky_relu(self.fc3(x))
        x = self.dropout(x)

        out = self.tahn(self.fc4(x))

        return out


def fake_loss(G_out: torch.Tensor):
    batch_size = G_out.size(0)
    labels = torch.ones(batch_size)

    return criterion(G_out.squeeze(), labels)

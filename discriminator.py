import torch
import torch.nn as nn

criterion = nn.BCEWithLogitsLoss()


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size):
        super(Discriminator, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_dim * 4)
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_size)

        self.dropout = nn.Dropout(p=0.3)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = self.leaky_relu(self.fc2(x))
        x = self.dropout(x)
        x = self.leaky_relu(self.fc3(x))
        x = self.dropout(x)

        out = self.leaky_relu(self.fc4(x))
        return out


def real_loss(D_out: torch.Tensor, cuda, smooth=False):
    batch_size = D_out.size(0)

    if smooth:
        labels = torch.ones(batch_size) * 0.9
    else:
        labels = torch.ones(batch_size)

    if cuda:
        labels = labels.cuda()

    return criterion(D_out.squeeze(), labels)

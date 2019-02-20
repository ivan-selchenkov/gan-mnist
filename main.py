import torch
from torchvision import datasets, transforms

from discriminator import Discriminator, real_loss
from generator import Generator, fake_loss

from torch import optim

num_workers = 0
batch_size = 64

transform = transforms.ToTensor()

train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)

lr=0.002

d_input_size = 28 * 28
d_hidden_dim = 32
d_output_size = 1

D = Discriminator(d_input_size, d_hidden_dim, d_output_size)
d_optim = optim.Adam(D.parameters(), lr=lr)

g_input_size = 100
g_hidden_dim = 32
g_output_size = 28 * 28

G = Generator(g_input_size, g_hidden_dim, g_output_size)
g_optim = optim.Adam(G.parameters(), lr=lr)



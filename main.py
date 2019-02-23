import torch
import pickle as pkl
from torchvision import datasets, transforms
from discriminator import Discriminator, real_loss
from generator import Generator, fake_loss
from torch import optim
from utils import generate_z_vector, generate_plot

import matplotlib.pyplot as plt

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

z_size = 100
g_hidden_dim = 32
g_output_size = 28 * 28

G = Generator(z_size, g_hidden_dim, g_output_size)
g_optim = optim.Adam(G.parameters(), lr=lr)

num_epochs = 100

samples = []
losses = []

print_every = 400

sample_size = 16

if torch.cuda.is_available():
    cuda = True
else:
    cuda = False

fixed_z = generate_z_vector(sample_size, z_size, cuda)

D.train()
G.train()

if cuda:
    D.cuda()
    G.cuda()


def train_discriminator(real_images, optimizer, batch_size, z_size):
    optimizer.zero_grad()

    if cuda:
        real_images = real_images.cuda()

    # Loss for real image
    d_real_loss = real_loss(D(real_images), cuda, smooth=True)

    # Loss for fake image
    fake_images = G(generate_z_vector(batch_size, z_size, cuda))
    d_fake_loss = fake_loss(D(fake_images), cuda)

    # add up loss and perform back-propagation
    d_loss = d_real_loss + d_fake_loss
    d_loss.backward()
    optimizer.step()

    return d_loss


def train_generator(optimizer, batch_size, z_size):
    optimizer.zero_grad()

    generated_images = G(generate_z_vector(batch_size, z_size, cuda))
    # training generator on real loss
    g_loss = real_loss(D(generated_images), cuda, smooth=True)
    g_loss.backward()
    optimizer.step()

    return g_loss


for epoch in range(num_epochs):
    for batch_i, (real_images, _) in enumerate(train_loader):
        batch_size = real_images.size(0)

        # transform real image data from [0, 1) to [-1, 1)
        real_images = real_images * 2 - 1

        d_loss = train_discriminator(real_images, d_optim, batch_size, z_size)
        g_loss = train_generator(g_optim, batch_size, z_size)

        # Print some loss stats
        if batch_i % print_every == 0:
            # print discriminator and generator loss
            print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                epoch + 1, num_epochs, d_loss.item(), g_loss.item()))

    losses.append((d_loss.item(), g_loss.item()))

    # generate and save sample, fake images
    G.eval()  # eval mode for generating samples
    samples_z = G(fixed_z)
    samples.append(samples_z)
    G.train()  # back to train mode

with open('train_samples.pkl', 'wb') as f:
    pkl.dump(samples, f)

generate_plot(losses)

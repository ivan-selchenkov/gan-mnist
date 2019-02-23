import numpy as np
import torch
import matplotlib.pyplot as plt


def generate_z_vector(sample_size, z_size, cuda):
    z_vectors = np.random.uniform(-1, 1, size=(sample_size, z_size))
    z_vectors = torch.from_numpy(z_vectors).float()

    if cuda:
        z_vectors = z_vectors.cuda()

    return z_vectors


def generate_plot(losses):
    fig, ax = plt.subplots()
    losses = np.array(losses)
    plt.plot(losses.T[0], label='Discriminator')
    plt.plot(losses.T[1], label='Generator')
    plt.title("Training Losses")
    plt.legend()
    plt.show()


def view_samples(epoch, samples):
    fig, axes = plt.subplots(figsize=(7,7), nrows=4, ncols=4, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples[epoch]):
        img = img.detach()
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(img.reshape((28,28)), cmap='Greys_r')
    plt.show()

import matplotlib.pyplot as plt
import numpy as np

losses = []
losses.append((1, 1))
losses.append((2, 4))
losses.append((3, 9))

fig, ax = plt.subplots()
losses = np.array(losses)
plt.plot(losses.T[0], label='Discriminator')
plt.plot(losses.T[1], label='Generator')
plt.title("Training Losses")
plt.legend()
plt.show()
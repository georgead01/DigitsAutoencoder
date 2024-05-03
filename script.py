from Autoencoder import Autoencoder

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import torch

digits = load_digits()
data = digits.data

n_samples, size = data.shape
size = int(np.sqrt(size))

data = torch.from_numpy(np.float32(data.reshape((n_samples, size, size))))

latent_size = 5

auto_enc = Autoencoder(size, latent_size)

x_train, x_test, y_train, y_test = train_test_split(data, data)

train_loss = auto_enc.train(x_train, y_train, 20)

if __name__ == '__main__':
    idx = np.random.choice(len(data))
    img = torch.from_numpy(np.float32(data[idx:idx+1]))

    latent_img = auto_enc.encode(img)
    reconstructed = auto_enc.decode(latent_img)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img.squeeze())
    axes[1].imshow(latent_img.detach().squeeze())
    axes[2].imshow(reconstructed.detach().squeeze())
    
    plt.tight_layout()
    plt.show() 

    plt.plot(np.arange(len(train_loss)), train_loss)
    plt.show()
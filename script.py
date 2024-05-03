from Autoencoder import Autoencoder

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import torch

digits = load_digits()
data = digits.data
labels = digits.target

n_samples, size = data.shape
size = int(np.sqrt(size))

data = torch.from_numpy(np.float32(data.reshape((n_samples, size, size))))

latent_size = 3

auto_enc = Autoencoder(size, latent_size)

x_train, x_test, y_train, y_test = train_test_split(data, labels)

train_loss = auto_enc.train(x_train, x_train, 20)
test_loss = auto_enc.test(x_test, x_test)
print(f'test loss: {test_loss}')

if __name__ == '__main__':
    latent_space = auto_enc.encode(data.unsqueeze(1))
    
    pca = PCA(2)
    latent_space = pca.fit_transform(latent_space.detach().reshape(n_samples, latent_size*latent_size))

    print(latent_space.shape)
    plt.scatter(latent_space[:,0], latent_space[:,1], c = labels)
    plt.show()

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
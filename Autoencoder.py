from typing import Any
from torch import nn, Size, optim
import math

class Autoencoder:
    def __init__(self, input_size, latent_size):

        self.latent_size = latent_size
        self.k_size = k_size = 3
        self.input_size = input_size

        post_conv_size = input_size-2*(k_size-1)

        self.model = nn.Sequential(
            nn.Conv2d(1, 16, k_size),
            nn.ReLU(),
            nn.Conv2d(16, 1, k_size),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(post_conv_size**2, latent_size*latent_size),
            nn.ReLU(),
            nn.Linear(latent_size*latent_size, post_conv_size**2),
            nn.ReLU(),
            nn.Unflatten(1, Size([post_conv_size,post_conv_size])),
            nn.ConvTranspose2d(1, 16, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3)
        )

        self.sep = 6
        self.encoder = self.model[:self.sep]
        self.decoder = self.model[self.sep:]
    
    def forward(self, inp):
        return self.model(inp)
    
    def encode(self, inp):
        out = self.encoder(inp)
        out = nn.Unflatten(1, [self.latent_size, self.latent_size])(out)
        return out
    
    def decode(self, inp):
        out = nn.Flatten()(inp)
        out = self.decoder(out)
        return out
    
    def train(self, data, targets, num_epochs = 10, lr = 0.001, optimizer = None):

        train_loss = []

        if not optimizer:
            optimizer = optim.SGD(self.model.parameters(), lr, weight_decay=0.1)

        self.optimizer = optimizer

        loss_fn = nn.MSELoss()

        for epoch in range(num_epochs):
            epoch_loss = 0
            for idx in range(len(data)):

                inp = data[idx:idx+1]
                target = targets[idx:idx+1]

                optimizer.zero_grad()

                out = self.forward(inp)
            
                loss = loss_fn(out, target)
                loss.backward()

                epoch_loss += loss.item()

                optimizer.step()

            epoch_loss /= len(data)
            train_loss.append(epoch_loss)

            print(f'epoch: {epoch}, loss: {epoch_loss}')

        return train_loss
  
if __name__ == '__main__':

    pass
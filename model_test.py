import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
import random

class Encoder(nn.Module):
    def __init__(self, lat_space):
        super().__init__() # 3,256,256
        '''TODO: Max, Average or Adaptive Pooling Layer; diluted convlayer?'''
        self.encoder_cnn = nn.Sequential(
            # [(W−K+2P)/S]+1
            nn.Conv2d(1, 64, 4, stride=2, padding=1), # 1x56x56 - 64x28x28
            nn.BatchNorm2d(64),
            nn.LeakyReLU(True),
            # [(Input size + 2 * Padding - dilation * (Kernel size - 1) - 1) / Stride] + 1
            nn.MaxPool2d(kernel_size=2), # 64x28x28 - 64x14x14
            nn.Conv2d(64, 32, 4, stride=2, padding=1), # 64x14x14 -32x7x7
            nn.BatchNorm2d(32), 
            nn.Dropout(0.5),
            # we want the encoding of each image to be independent of all the other images
            nn.LeakyReLU(True),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),  #32x7x7 - 32x7x7 I add this layer to increase the complexity
            nn.BatchNorm2d(32),
            nn.Dropout(0.5),
            nn.LeakyReLU(True),
            nn.MaxPool2d(kernel_size=2), # 32x7x7 - 32x3x3
            nn.Conv2d(32, 16, 4, stride=2, padding=1), # 32x3x3 -16x1x1
            nn.BatchNorm2d(16),
            nn.LeakyReLU(True)
            # nn.Conv2d(16, 8, 4, stride=1, padding=1), 
            # nn.LeakyReLU(True)
        )



        self.flatten = nn.Flatten(start_dim=1)

        self.encoder_lin = nn.Sequential(
            # nn.Linear(16*1*1, 56),
            # nn.LeakyReLU(True),
            nn.Linear(16*1*1, lat_space)
        )

    def forward(self, x):
        x = self.encoder_cnn(x)
        # x = x.view(x.size(0), -1)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x


class Decoder(nn.Module):

    def __init__(self,lat_space):
        super().__init__()

        self.decoder_lin = nn.Sequential(
            nn.Linear(lat_space, 16),
            nn.LeakyReLU(True),
            # nn.Linear(56, 16),
            # nn.LeakyReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(16, 1, 1))

        self.decoder_conv = nn.Sequential(
            #nn.ConvTranspose2d(8, 16, 4, stride=2, padding=1, output_padding=0),
            # nn.BatchNorm2d(16),
            
            nn.ConvTranspose2d(16, 32, 4, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.Dropout(0.5),
            nn.LeakyReLU(True),
            nn.Upsample(size=7),
            nn.ConvTranspose2d(32, 32, 3, stride=1, padding=1, output_padding=0),
            nn.BatchNorm2d(32),
            nn.Dropout(0.5),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(32, 64, 4, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(64),
            # nn.BatchNorm2d(8),
            nn.LeakyReLU(True),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1, output_padding=0),

        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.tanh(x)
        return x


class CNN_AE(nn.Module):
    def __init__(self, lat_space, device) -> None:
        super().__init__()

        #self.save_hyperparameters()
        
        
        self.device = device
        self.encoder = Encoder(lat_space).to(device)
        self.decoder = Decoder(lat_space).to(device)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def plot_outputs(self, vset, num_images):
        _, axs = plt.subplots(2, num_images)

        for i in range(num_images):
            idx = random.randint(0, len(vset))
            img = vset[idx][0].unsqueeze(0).to(self.device)
            axs[0, i].imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
            axs[0, i].get_xaxis().set_visible(False)
            axs[0, i].get_yaxis().set_visible(False)
            if i == num_images//2:
                axs[0, i].set_title('Original images')

            self.eval()
            with torch.no_grad():
                rec_img = self.decoder(self.encoder(img))
            axs[1, i].imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')
            axs[1, i].get_xaxis().set_visible(False)
            axs[1, i].get_yaxis().set_visible(False)
            if i == num_images//2:
                axs[1, i].set_title('Reconstructed images')
        plt.show()
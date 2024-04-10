import torch 
from torch import nn 
import torch.nn.functional as F
from matplotlib import pyplot as plt
import random


class Encoder(nn.Module):
    def __init__(self, lat_space, variational=False):
        super().__init__()
        self.variational = variational
        self.dropout = 0.5
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(self.dropout)
        #[(W−K+2P)/S]+1
        #block conv1
        self.bn1 = nn.BatchNorm2d(64)
        self.conv1 = nn.Conv2d(1, 64, 7, stride=1, padding=3) # (3, 64, 7, stride=2, padding=3) -> (3, 64, 7, stride=1, padding=3)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        #block conv2x conv
        self.bn2 = nn.BatchNorm2d(64)

        self.conv2_conv_1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv2_conv_2 = nn.Conv2d(64, 64, 3, 1, 1)
        #block conv2x identity
        self.conv2_id_1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv2_id_2 = nn.Conv2d(64, 64, 3, 1, 1)
        #block conv3x conv
        self.bn3 = nn.BatchNorm2d(128)

        self.conv3_conv_1 = nn.Conv2d(64, 128, 3, 1, 1) # (64, 128, 3, 2, 1) -> (64, 128, 3, 1, 1)
        self.conv3_conv_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv3_resconv = nn.Conv2d(64, 128, 1, 1, 0) # (64, 128, 1, 2, 0) -> (64, 128, 1, 1, 0)
        #block conv3x identity
        self.conv3_id_1 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv3_id_2 = nn.Conv2d(128, 128, 3, 1, 1)
        #block conv4x conv
        self.bn4 = nn.BatchNorm2d(256)

        self.conv4_conv_1 = nn.Conv2d(128, 256, 3, 2, 1)
        self.conv4_conv_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv4_resconv = nn.Conv2d(128, 256, 1, 2, 0)
        #block conv4x identity
        self.conv4_id_1 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv4_id_2 = nn.Conv2d(256, 256, 3, 1, 1)
        #block conv5x conv
        self.bn5 = nn.BatchNorm2d(512)
        
        self.conv5_conv_1 = nn.Conv2d(256, 512, 3, 2, 1)
        self.conv5_conv_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv5_resconv = nn.Conv2d(256, 512, 1, 2, 0)
        #block con5x identity
        self.conv5_id_1 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv5_id_2 = nn.Conv2d(512, 512, 3, 1, 1)

        self.avgpool = nn.AvgPool2d(7) # some forums say that this should be adaptive avg pooling,
        self.flat = nn.Flatten()
        self.bottleneck = nn.Linear(1*1*512, lat_space)

        self.mean_layer = nn.Linear(512, lat_space)
        self.logvar_layer = nn.Linear(512, lat_space)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        h = self.maxpool(x)

        x = self.relu(self.bn2(self.conv2_conv_1(h)))
        x = self.bn2(self.conv2_conv_2(x))
        x = self.dropout(x)
        h = self.relu(x + h)

        x = self.relu(self.bn2(self.conv2_id_1(h)))
        x = self.bn2(self.conv2_id_2(x))
        x = self.dropout(x)
        h = self.relu(x + h)

        x = self.relu(self.bn3(self.conv3_conv_1(h)))
        x = self.bn3(self.conv3_conv_2(x))
        x = self.dropout(x)
        h = self.conv3_resconv(h)
        h = self.relu(x + h)

        x = self.relu(self.bn3(self.conv3_id_1(h)))
        x = self.bn3(self.conv3_id_2(x))
        x = self.dropout(x)
        h = self.relu(x + h)

        x = self.relu(self.bn4(self.conv4_conv_1(h)))
        x = self.bn4(self.conv4_conv_2(x))
        x = self.dropout(x)
        h = self.conv4_resconv(h)
        h = self.relu(x + h)

        x = self.relu(self.bn4(self.conv4_id_1(h)))
        x = self.bn4(self.conv4_id_2(x))
        x = self.dropout(x)
        h = self.relu(x + h)

        x = self.relu(self.bn5(self.conv5_conv_1(h)))
        x = self.bn5(self.conv5_conv_2(x))
        x = self.dropout(x)
        h = self.conv5_resconv(h)
        h = self.relu(x + h)

        x = self.relu(self.bn5(self.conv5_id_1(h)))
        x = self.bn5(self.conv5_id_2(x))
        x = self.dropout(x)
        h = self.relu(x + h)

        x = self.flat(self.avgpool(h))

        if self.variational:
            mean, logvar = self.mean_layer(x), self.logvar_layer(x)
            return mean, logvar
        return self.bottleneck(x)
    

class Decoder(nn.Module):
    def __init__(self, lat_space, variational=False):
        super().__init__()
        self.variational = variational
        self.dropout = 0.5
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(self.dropout)

        if variational:
            self.last = nn.Sequential(
            # nn.Linear(2, lat_space),
            # nn.ReLU(True),
            nn.Linear(lat_space, 512),
            nn.Unflatten(1, (512, 1, 1))
        )
        else:
            self.last = nn.Sequential(
                nn.Linear(lat_space, 512),
                nn.Unflatten(1, (512, 1, 1))
            )
        self.bn5 = nn.BatchNorm2d(512)
        self.deconv5_id_2 = nn.ConvTranspose2d(512, 512, 3, 1, 1)
        self.deconv5_id_1 = nn.ConvTranspose2d(512, 512, 3, 1, 1)

        self.conv5_resconv = nn.ConvTranspose2d(512, 256, 1, 2, 0, 1)
        self.deconv5_conv_2 = nn.ConvTranspose2d(512, 512, 3, 1, 1)
        self.deconv5_conv_1 = nn.ConvTranspose2d(512, 256, 3, 2, 1, 1) # output padding to get the mirrored size

        self.bn4 = nn.BatchNorm2d(256)
        self.deconv4_id_2 = nn.ConvTranspose2d(256, 256, 3, 1, 1)
        self.deconv4_id_1 = nn.ConvTranspose2d(256, 256, 3, 1, 1)

        self.conv4_resconv = nn.ConvTranspose2d(256, 128, 1, 2, 0, 1)
        self.deconv4_conv_2 = nn.ConvTranspose2d(256, 256, 3, 1, 1)
        self.deconv4_conv_1 = nn.ConvTranspose2d(256, 128, 3, 2, 1, 1) # output padding

        self.bn3 = nn.BatchNorm2d(128)
        self.deconv3_id_2 = nn.ConvTranspose2d(128, 128, 3, 1, 1)
        self.deconv3_id_1 = nn.ConvTranspose2d(128, 128, 3, 1, 1)

        self.conv3_resconv = nn.ConvTranspose2d(128, 64, 1, 1, 0)
        self.deconv3_conv_2 = nn.ConvTranspose2d(128, 128, 3, 1, 1)
        self.deconv3_conv_1 = nn.ConvTranspose2d(128, 64, 3, 1, 1) # output padding

        self.bn2 = nn.BatchNorm2d(64)
        self.deconv2_id_2 = nn.ConvTranspose2d(64, 64, 3, 1, 1)
        self.deconv2_id_1 = nn.ConvTranspose2d(64, 64, 3, 1, 1)

        self.deconv2_conv_2 = nn.ConvTranspose2d(64, 64, 3, 1, 1)
        self.deconv2_conv_1 = nn.ConvTranspose2d(64, 64, 3, 1, 1)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.deconv1 = nn.ConvTranspose2d(64, 1, 7, 1, 3)
        self.bn1 = nn.BatchNorm2d(64)

    def forward(self, x):
        h = self.last(x)
        h = F.interpolate(h, scale_factor=7, mode='nearest')
        x = self.relu(self.bn5(self.deconv5_id_2(h)))
        x = self.bn5(self.deconv5_id_1(x))
        x = self.dropout(x)
        h = self.relu(x + h)

        x = self.relu(self.bn5(self.deconv5_conv_2(h)))
        x = self.bn4(self.deconv5_conv_1(x))
        x = self.dropout(x)
        h = self.conv5_resconv(h)
        h = self.relu(x + h)

        x = self.relu(self.bn4(self.deconv4_id_2(h)))
        x = self.bn4(self.deconv4_id_1(x))
        x = self.dropout(x)
        h = self.relu(x + h)

        x = self.relu(self.bn4(self.deconv4_conv_2(h)))
        x = self.bn3(self.deconv4_conv_1(x))
        x = self.dropout(x)
        h = self.conv4_resconv(h)
        h = self.relu(x + h)
        
        x = self.relu(self.bn3(self.deconv3_id_2(h)))
        x = self.bn3(self.deconv3_id_1(x))
        x = self.dropout(x)
        h = self.relu(x + h)

        x = self.relu(self.bn3(self.deconv3_conv_2(h)))
        x = self.bn2(self.deconv3_conv_1(x))
        x = self.dropout(x)
        h = self.conv3_resconv(h)
        h = self.relu(x + h)

        x = self.relu(self.bn2(self.deconv2_id_2(h)))
        x = self.bn2(self.deconv2_id_1(x))
        x = self.dropout(x)
        h = self.relu(x + h)

        x = self.relu(self.bn2(self.deconv2_conv_2(h)))
        x = self.bn1(self.deconv2_conv_1(x))
        x = self.dropout(x)
        h = self.relu(x + h)

        x = self.deconv1(self.upsample(h))
        x = torch.sigmoid(x)
        #x = x.view(x.size(0), 1, 56, 56)
        return x
    

class AE(nn.Module):
    def __init__(self, lat_space, device, variational=False) -> None:
        super().__init__()
        
        self.encoder = Encoder(lat_space, variational).to(device)
        self.decoder = Decoder(lat_space, variational).to(device)
        self.device = device
        self.variational = variational

    def forward(self, x):
        if self.variational:
            m, v = self.encoder(x)
            z = self.reparam(m, v)
            x = self.decoder(z)
            return x, m, v
        else:
            x = self.encoder(x)
            x = self.decoder(x)
            return x
    
    def reparam(self, mean, var):
        eps = torch.randn_like(var).to(self.device)
        z = mean + var * eps
        return z
    
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







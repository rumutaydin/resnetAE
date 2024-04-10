class ResNet18VAEEncoder(nn.Module):
    def _init_(self, latent_features):
        super(ResNet18VAEEncoder, self)._init_()
        
        self.dropout_percentage = 0.5
        self.relu = nn.ReLU()
        
        # BLOCK-1 
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(7,7), stride=(2,2), padding=(3,3))
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=(1,1))
        
        # BLOCK-2 
        self.conv2_1_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm2_1_1 = nn.BatchNorm2d(64)
        self.conv2_1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm2_1_2 = nn.BatchNorm2d(64)
        self.dropout2_1 = nn.Dropout(p=self.dropout_percentage)
        # BLOCK-2 (2)
        self.conv2_2_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm2_2_1 = nn.BatchNorm2d(64)
        self.conv2_2_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm2_2_2 = nn.BatchNorm2d(64)
        self.dropout2_2 = nn.Dropout(p=self.dropout_percentage)
        
        # BLOCK-3 (1) 
        self.conv3_1_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=(2,2), padding=(1,1))
        self.batchnorm3_1_1 = nn.BatchNorm2d(128)
        self.conv3_1_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm3_1_2 = nn.BatchNorm2d(128)
        self.concat_adjust_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1,1), stride=(2,2), padding=(0,0))
        self.dropout3_1 = nn.Dropout(p=self.dropout_percentage)
        # BLOCK-3 (2)
        self.conv3_2_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm3_2_1 = nn.BatchNorm2d(128)
        self.conv3_2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm3_2_2 = nn.BatchNorm2d(128)
        self.dropout3_2 = nn.Dropout(p=self.dropout_percentage)
        
        # BLOCK-4 (1) 
        self.conv4_1_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), stride=(2,2), padding=(1,1))
        self.batchnorm4_1_1 = nn.BatchNorm2d(256)
        self.conv4_1_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm4_1_2 = nn.BatchNorm2d(256)
        self.concat_adjust_4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1,1), stride=(2,2), padding=(0,0))
        self.dropout4_1 = nn.Dropout(p=self.dropout_percentage)
        # BLOCK-4 (2)
        self.conv4_2_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm4_2_1 = nn.BatchNorm2d(256)
        self.conv4_2_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm4_2_2 = nn.BatchNorm2d(256)
        self.dropout4_2 = nn.Dropout(p=self.dropout_percentage)
        
        # BLOCK-5 (1)
        self.conv5_1_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), stride=(2,2), padding=(1,1))
        self.batchnorm5_1_1 = nn.BatchNorm2d(512)
        self.conv5_1_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm5_1_2 = nn.BatchNorm2d(512)
        self.concat_adjust_5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(1,1), stride=(2,2), padding=(0,0))
        self.dropout5_1 = nn.Dropout(p=self.dropout_percentage)
        # BLOCK-5 (2)
        self.conv5_2_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm5_2_1 = nn.BatchNorm2d(512)
        self.conv5_2_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm5_2_2 = nn.BatchNorm2d(512)
        self.dropout5_2 = nn.Dropout(p=self.dropout_percentage)
        
        # Final Block  
        self.sampling = Sampling()
        self.fc_mu = nn.Linear(in_features=512, out_features=latent_features)  # Output layer for mean
        self.fc_logvar = nn.Linear(in_features=512, out_features=latent_features)  # Output layer for log-variance
     
    def forward(self, x):
        
        # Block 1
        x = self.relu(self.batchnorm1(self.conv1(x)))
        x = self.maxpool1(x)
        
        # Block 2
        x1 = self.relu(self.batchnorm2_1_1(self.conv2_1_1(x)))
        x1 = self.batchnorm2_1_2(self.conv2_1_2(x1))
        x1 = self.dropout2_1(x1)
        x1 = self.relu(x1 + x)
        x2 = self.relu(self.batchnorm2_2_1(self.conv2_2_1(x1)))
        x2 = self.batchnorm2_2_2(self.conv2_2_2(x2))
        x2 = self.dropout2_2(x2)
        x = self.relu(x2 + x1)
        
        # Block 3
        x1 = self.relu(self.batchnorm3_1_1(self.conv3_1_1(x)))
        x1 = self.batchnorm3_1_2(self.conv3_1_2(x1))
        x1 = self.dropout3_1(x1)
        x1 = self.relu(x1 + self.concat_adjust_3(x))
        x2 = self.relu(self.batchnorm3_2_1(self.conv3_2_1(x1)))
        x2 = self.batchnorm3_2_2(self.conv3_2_2(x2))
        x2 = self.dropout3_2(x2)
        x = self.relu(x2 + x1)
        
        # Block 4
        x1 = self.relu(self.batchnorm4_1_1(self.conv4_1_1(x)))
        x1 = self.batchnorm4_1_2(self.conv4_1_2(x1))
        x1 = self.dropout4_1(x1)
        x1 = self.relu(x1 + self.concat_adjust_4(x))
        x2 = self.relu(self.batchnorm4_2_1(self.conv4_2_1(x1)))
        x2 = self.batchnorm4_2_2(self.conv4_2_2(x2))
        x2 = self.dropout4_2(x2)
        x = self.relu(x2 + x1)
        
        # Block 5
        x1 = self.relu(self.batchnorm5_1_1(self.conv5_1_1(x)))
        x1 = self.batchnorm5_1_2(self.conv5_1_2(x1))
        x1 = self.dropout5_1(x1)
        x1 = self.relu(x1 + self.concat_adjust_5(x))
        x2 = self.relu(self.batchnorm5_2_1(self.conv5_2_1(x1)))
        x2 = self.batchnorm5_2_2(self.conv5_2_2(x2))
        x2 = self.dropout5_2(x2)
        x = self.relu(x2 + x1)
        
        # Final block - Global Average Pooling
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        
        # Output layer for mean
        mu = self.fc_mu(x)
        
        # Output layer for log-variance
        logvar = self.fc_logvar(x)
        
        z = self.sampling(mu, logvar)
        
        return mu, logvar,z
    
    
class ResNet18VAEDecoder(nn.Module):
    def _init_(self, latent_features):
        super(ResNet18VAEDecoder, self)._init_()
        
        self.dropout_percentage = 0.5
        self.relu = nn.ReLU()
        
        # Reverse of Final Block in Encoder
        self.fc = nn.Linear(in_features=latent_features, out_features=512)
        
        # Reverse of Block-5
        self.upsample5 = nn.Upsample(scale_factor=2)
        self.conv5_2_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm5_2_2 = nn.BatchNorm2d(512)
        self.conv5_2_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm5_2_1 = nn.BatchNorm2d(512)
        self.dropout5_2 = nn.Dropout(p=self.dropout_percentage)
        self.conv5_1_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm5_1_2 = nn.BatchNorm2d(512)
        self.conv5_1_1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm5_1_1 = nn.BatchNorm2d(256)
        
        # Reverse of Block-4
        self.upsample4 = nn.Upsample(scale_factor=2)
        self.conv4_2_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm4_2_2 = nn.BatchNorm2d(256)
        self.conv4_2_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm4_2_1 = nn.BatchNorm2d(256)
        self.dropout4_2 = nn.Dropout(p=self.dropout_percentage)
        self.conv4_1_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm4_1_2 = nn.BatchNorm2d(256)
        self.conv4_1_1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm4_1_1 = nn.BatchNorm2d(128)
        
        # Reverse of Block-3
        self.upsample3 = nn.Upsample(scale_factor=2)
        self.conv3_2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm3_2_2 = nn.BatchNorm2d(128)
        self.conv3_2_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm3_2_1 = nn.BatchNorm2d(128)
        self.dropout3_2 = nn.Dropout(p=self.dropout_percentage)
        self.conv3_1_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm3_1_2 = nn.BatchNorm2d(128)
        self.conv3_1_1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm3_1_1 = nn.BatchNorm2d(64)
        
        # Reverse of Block-2
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.conv2_2_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm2_2_2 = nn.BatchNorm2d(64)
        self.conv2_2_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm2_2_1 = nn.BatchNorm2d(64)
        self.dropout2_2 = nn.Dropout(p=self.dropout_percentage)
        self.conv2_1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm2_1_2 = nn.BatchNorm2d(64)
        self.conv2_1_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm2_1_1 = nn.BatchNorm2d(64)
        
        # Reverse of Block-1
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.conv1_1 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm1_1 = nn.BatchNorm2d(1)

    def forward(self, x):
        
        # Final Block
        x = self.relu(self.fc(x))
        x = x.view(-1, 512, 1, 1)  # Reshape to match dimensions
        
        # Block 5
        x = self.relu(self.batchnorm5_2_2(self.conv5_2_2(self.upsample5(x))))
        x = self.relu(self.batchnorm5_2_1(self.conv5_2_1(x)))
        x = self.dropout5_2(x)
        x = self.relu(self.batchnorm5_1_2(self.conv5_1_2(x)))
        x = self.relu(self.batchnorm5_1_1(self.conv5_1_1(x)))
        
        # Block 4
        x = self.relu(self.batchnorm4_2_2(self.conv4_2_2(self.upsample4(x))))
        x = self.relu(self.batchnorm4_2_1(self.conv4_2_1(x)))
        x = self.dropout4_2(x)
        x = self.relu(self.batchnorm4_1_2(self.conv4_1_2(x)))
        x = self.relu(self.batchnorm4_1_1(self.conv4_1_1(x)))
       
        # Block 3
        x = self.relu(self.batchnorm3_2_2(self.conv3_2_2(self.upsample3(x))))
        x = self.relu(self.batchnorm3_2_1(self.conv3_2_1(x)))
        x = self.dropout3_2(x)
        x = self.relu(self.batchnorm3_1_2(self.conv3_1_2(x)))
        x = self.relu(self.batchnorm3_1_1(self.conv3_1_1(x)))
        
        # Block 2
        x = self.relu(self.batchnorm2_2_2(self.conv2_2_2(self.upsample2(x))))
        x = self.relu(self.batchnorm2_2_1(self.conv2_2_1(x)))
        x = self.dropout2_2(x)
        x = self.relu(self.batchnorm2_1_2(self.conv2_1_2(x)))
        x = self.relu(self.batchnorm2_1_1(self.conv2_1_1(x)))
        
        # Block 1
        x = self.relu(self.batchnorm1_1(self.conv1_1(self.upsample1(x))))
        
        return x
    

class VAE(nn.Module):
    def _init_(self, encoder, decoder):
        super(VAE, self)._init_()
        # initialize the encoder and decoder
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, x):
        # pass the input through the encoder to get the latent vector
        z_mean, z_log_var, z = self.encoder(x)
        
        # pass the latent vector through the decoder to get the reconstructed
        # image
        reconstruction = self.decoder(z)
        # return the mean, log variance and the reconstructed image
        return z_mean, z_log_var, reconstruction
import torch
import torch.nn as nn
import torch.nn.functional as F


# Define a basic 3D convolutional block
class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs=2, kernel_size=3, norm_layer=nn.InstanceNorm3d, nonlin=nn.LeakyReLU):
        super(ConvBlock3D, self).__init__()
        layers = []
        for i in range(num_convs):
            layers.append(nn.Conv3d(in_channels if i == 0 else out_channels, out_channels, kernel_size=kernel_size, padding=1, bias=True))
            layers.append(norm_layer(out_channels, eps=1e-5, affine=True))
            layers.append(nonlin(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

# Define the UNet model to match the nnUNet 3d_fullres configuration
class nnUNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=3):
        super(nnUNet3D, self).__init__()
        #features = [32, 64, 128, 256, 320]
        features = [32, 64, 128, 256, 512]

        # Encoder
        self.encoder1 = ConvBlock3D(in_channels, features[0])
        self.down1 = nn.Conv3d(features[0], features[0], kernel_size=3, stride=2, padding=1)  # Stride [2,2,2]

        self.encoder2 = ConvBlock3D(features[0], features[1])
        self.down2 = nn.Conv3d(features[1], features[1], kernel_size=3, stride=2, padding=1)

        self.encoder3 = ConvBlock3D(features[1], features[2])
        self.down3 = nn.Conv3d(features[2], features[2], kernel_size=3, stride=2, padding=1)

        self.encoder4 = ConvBlock3D(features[2], features[3])
        self.down4 = nn.Conv3d(features[3], features[3], kernel_size=3, stride=2, padding=1)

        self.bottleneck = ConvBlock3D(features[3], features[4])

        # Decoder
        self.up4 = nn.ConvTranspose3d(features[4], features[3], kernel_size=2, stride=2)
        self.decoder4 = ConvBlock3D(features[4], features[3])

        self.up3 = nn.ConvTranspose3d(features[3], features[2], kernel_size=2, stride=2)
        self.decoder3 = ConvBlock3D(features[3], features[2])

        self.up2 = nn.ConvTranspose3d(features[2], features[1], kernel_size=2, stride=2)
        self.decoder2 = ConvBlock3D(features[2], features[1])

        self.up1 = nn.ConvTranspose3d(features[1], features[0], kernel_size=2, stride=2)
        self.decoder1 = ConvBlock3D(features[1], features[0])

        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        x = self.down1(enc1)

        enc2 = self.encoder2(x)
        x = self.down2(enc2)

        enc3 = self.encoder3(x)
        x = self.down3(enc3)

        enc4 = self.encoder4(x)
        x = self.down4(enc4)

        x = self.bottleneck(x)

        x = self.up4(x)
        x = torch.cat((x, enc4), dim=1)
        x = self.decoder4(x)

        x = self.up3(x)
        x = torch.cat((x, enc3), dim=1)
        x = self.decoder3(x)

        x = self.up2(x)
        x = torch.cat((x, enc2), dim=1)
        x = self.decoder2(x)

        x = self.up1(x)
        x = torch.cat((x, enc1), dim=1)
        x = self.decoder1(x)

        out = self.final_conv(x)
        #out = F.softmax(out, dim=1)
        return out, enc4




# Define the UNet model to match the nnUNet 3d_fullres configuration
class nnUNet3D_Dropout(nn.Module):
    def __init__(self, in_channels=1, out_channels=3):
        super(nnUNet3D_Dropout, self).__init__()
        #features = [32, 64, 128, 256, 320]
        features = [32, 64, 128, 256, 512]

        # Encoder
        self.encoder1 = ConvBlock3D(in_channels, features[0])
        self.down1 = nn.Conv3d(features[0], features[0], kernel_size=3, stride=2, padding=1)  # Stride [2,2,2]

        self.encoder2 = ConvBlock3D(features[0], features[1])
        self.down2 = nn.Conv3d(features[1], features[1], kernel_size=3, stride=2, padding=1)

        self.encoder3 = ConvBlock3D(features[1], features[2])
        self.down3 = nn.Conv3d(features[2], features[2], kernel_size=3, stride=2, padding=1)

        self.encoder4 = ConvBlock3D(features[2], features[3])
        self.down4 = nn.Conv3d(features[3], features[3], kernel_size=3, stride=2, padding=1)

        self.bottleneck = ConvBlock3D(features[3], features[4])

        # Decoder
        self.up4 = nn.ConvTranspose3d(features[4], features[3], kernel_size=2, stride=2)
        self.decoder4 = ConvBlock3D(features[4], features[3])

        self.up3 = nn.ConvTranspose3d(features[3], features[2], kernel_size=2, stride=2)
        self.decoder3 = ConvBlock3D(features[3], features[2])

        self.up2 = nn.ConvTranspose3d(features[2], features[1], kernel_size=2, stride=2)
        self.decoder2 = ConvBlock3D(features[2], features[1])

        self.up1 = nn.ConvTranspose3d(features[1], features[0], kernel_size=2, stride=2)
        self.decoder1 = ConvBlock3D(features[1], features[0])

        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        x = self.down1(enc1)

        enc2 = self.encoder2(x)
        x = self.down2(enc2)

        enc3 = self.encoder3(x)
        x = self.down3(enc3)

        enc4 = self.encoder4(x)
        x = self.down4(enc4)

        x = self.bottleneck(x)

        x = self.up4(x)
        x = torch.cat((x, enc4), dim=1)
        x = self.decoder4(x)

        x = self.up3(x)
        x = torch.cat((x, enc3), dim=1)
        x = self.decoder3(x)

        x = self.up2(x)
        x = torch.cat((x, enc2), dim=1)
        x = self.decoder2(x)

        x = self.up1(x)
        x = torch.cat((x, enc1), dim=1)
        x = self.decoder1(x)

        out = self.final_conv(x)
        #out = F.softmax(out, dim=1)
        return out, enc4





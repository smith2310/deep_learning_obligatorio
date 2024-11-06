import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    
    def __init__(self, input_channel, out_channel, starting_channel=64):
        super(UNet, self).__init__()
        self.enc1 = self.conv_block(input_channel, starting_channel)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self.conv_block(starting_channel, starting_channel*2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = self.conv_block(starting_channel*2, starting_channel*4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = self.conv_block(starting_channel*4, starting_channel*8)
        self.pool4 = nn.MaxPool2d(2)
        self.base = self.conv_block(starting_channel*8, starting_channel*16)
        self.up4 = nn.ConvTranspose2d(starting_channel*16, starting_channel*8, 2, stride=2)
        self.dec4 = self.conv_block(starting_channel*16, starting_channel*8)
        self.up3 = nn.ConvTranspose2d(starting_channel*8, starting_channel*4, 2, stride=2)
        self.dec3 = self.conv_block(starting_channel*8, starting_channel*4)
        self.up2 = nn.ConvTranspose2d(starting_channel*4, starting_channel*2, 2, stride=2)
        self.dec2 = self.conv_block(starting_channel*4, starting_channel*2)
        self.up1 = nn.ConvTranspose2d(starting_channel*2, starting_channel, 2, stride=2)
        self.dec1 = self.conv_block(starting_channel*2, starting_channel)
        self.output = nn.Conv2d(starting_channel, out_channel, 1)
        self.output_activation = nn.Sigmoid()
        

        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        enc1 = self.enc1(x)
        pool1 = self.pool1(enc1)
        enc2 = self.enc2(pool1)
        pool2 = self.pool2(enc2)
        enc3 = self.enc3(pool2)
        pool3 = self.pool3(enc3)
        enc4 = self.enc4(pool3)
        pool4 = self.pool4(enc4)
        base = self.base(pool4)
        up4 = self.up4(base)
        dec4 = self.dec4(torch.cat([up4, enc4], 1))
        up3 = self.up3(dec4)
        dec3 = self.dec3(torch.cat([up3, enc3], 1))
        up2 = self.up2(dec3)
        dec2 = self.dec2(torch.cat([up2, enc2], 1))
        up1 = self.up1(dec2)
        dec1 = self.dec1(torch.cat([up1, enc1], 1))
        output = self.output(dec1)
        return self.output_activation(output)

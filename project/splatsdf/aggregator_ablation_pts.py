import torch
import torch.nn as nn
    

class Aggregator(torch.nn.Module):

    def __init__(self):
        super(Aggregator, self).__init__()
        
        
        out_channels = 128
        in_channels = 128
        block1 = []
        for i in range(2):
            block1.append(nn.Linear(in_channels, out_channels))
            block1.append(nn.LeakyReLU(inplace=True))
            in_channels = out_channels
        self.block1 = nn.Sequential(*block1)

        in_channels = 128
        block2 = []
        for i in range(2):
            block2.append(nn.Linear(in_channels, out_channels))
            block2.append(nn.LeakyReLU(inplace=True))
            in_channels = out_channels
        self.block2 = nn.Sequential(*block2)


        in_channels = 128
        block3 = []
        for i in range(2):
            block3.append(nn.Linear(in_channels, out_channels))
            block3.append(nn.Tanh())
            in_channels = out_channels
        self.block3 = nn.Sequential(*block3)

        
    def forward(self, sampled_rgb, sampled_sh, sampled_embedding, cov3d):
        
        # no positional encoding because it's already done in hash encoding section   
        feat = sampled_embedding
        feat = self.block1(feat)
        feat = self.block2(feat)
        feat = self.block3(feat)
        
        return feat
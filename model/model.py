import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from .ResNeST import resnest50

class DecoderBlock(nn.Module):
    def __init__(self, ch_in, ch_mid, ch_out, k, s, p, op=0):
        super(DecoderBlock, self).__init__()

        '''Transposed Conv Block'''
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_mid, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=ch_mid, out_channels=ch_out, kernel_size=k,
                               stride=s, padding=p, output_padding=op),
        )

    def forward(self, x):
        x = self.block(x)
        return x

class Decoder_ResNeSt(nn.Module):

    def __init__(self):
        super(Decoder_ResNeSt, self).__init__()

        '''Instance Segmentation Method'''
        self.dec5 = DecoderBlock(ch_in=2048, ch_mid=64, ch_out=1024, k=4, s=2, p=1)  
        self.dec4 = DecoderBlock(ch_in=1024 * 2, ch_mid=64, ch_out=512, k=4, s=2, p=1)
        self.dec3 = DecoderBlock(ch_in=512 * 2, ch_mid=64, ch_out=256, k=4, s=2, p=1)
        self.dec2 = DecoderBlock(ch_in=256 * 2, ch_mid=64, ch_out=64, k=4, s=2, p=1)
        self.dec1 = DecoderBlock(ch_in=64 * 2, ch_mid=64, ch_out=64, k=4, s=2, p=1)
        self.conv_logit = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1)
        self.conv_embedding = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, input_tensor_list):
        """
        :param input_tensor_list:
        :return:
        """

        dec5 = self.dec5(input_tensor_list[4])
        # print(dec5.shape)
        dec4 = self.dec4(torch.cat((dec5, input_tensor_list[3]), 1))
        # print(dec4.shape)
        dec3 = self.dec3(torch.cat((dec4, input_tensor_list[2]), 1))
        # print(dec3.shape)
        dec2 = self.dec2(torch.cat((dec3, input_tensor_list[1]), 1))
        # print(dec2.shape)
        dec1 = self.dec1(torch.cat((dec2, input_tensor_list[0]), 1))
        # print(dec1.shape)

        logit = self.conv_logit(dec1)
        embedding = self.conv_embedding(dec1)

        return embedding, logit


class Resnest_LaneNet(nn.Module):

    def __init__(self):

        super().__init__()

        # comment or uncomment to choose from different encoders and decoders
        self.encoder = resnest50(pretrained=False)
        self.decoder = Decoder_ResNeSt()  # Decoder with Transposed Conv
        

    def forward(self, input):
        x = self.encoder.forward(input)
        
        # store feature maps of the encoder for later fusion in the decoder
        input_tensor_list = [self.encoder.c1, self.encoder.c2, self.encoder.c3, self.encoder.c4, x]
        embedding,logit = self.decoder.forward(input_tensor_list)

        return embedding, logit, input_tensor_list

# from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    model = Resnest_LaneNet()
    my_input = torch.rand(3, 3, 288, 512)
    out = model(my_input)
    total = sum([param.nelement() for param in model.parameters()])
    print(total)
   
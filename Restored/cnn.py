import torch
import torch.nn as nn
# from python_files.utils import *
from utils import *
# import torchsummary

# class EncoderCNN(nn.Module):
#     def __init__(self) -> None:
#         super(EncoderCNN, self).__init__()
        
#         # input: 30 x 1200  output: 373 x 200
#         self.convSequence = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=(5, 20), stride=(1, 1), padding=(2, 0)),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),

#             nn.Conv2d(32, 32, kernel_size=(5, 20), stride=(1, 1), padding=(2, 0)), 
#             nn.BatchNorm2d(32),
#             nn.ReLU(),

#             nn.Conv2d(32, 64, kernel_size=(5, 20), stride=(1, 1), padding=(2, 0)),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),

#             nn.Conv2d(64, 64, kernel_size=(5, 20), stride=(1, 1), padding=(0, 0)),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),

#             nn.Conv2d(64, 128, kernel_size=(5, 20), stride=(1, 1), padding=(2, 0)),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),

#             nn.Conv2d(128, 128, kernel_size=(5, 20), stride=(1, 1), padding=(2, 0)),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),

#             nn.Conv2d(128, 256, kernel_size=(5, 20), stride=(1, 1), padding=(0, 0)),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),

#             nn.Conv2d(256, 256, kernel_size=(5, 20), stride=(1, 1), padding=(0, 0)),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),

#             nn.Conv2d(256, 512, kernel_size=(5, 20), stride=(1, 1), padding=(0, 0)),
#             nn.BatchNorm2d(512),
#             nn.ReLU(),

#             nn.Conv2d(512, 512, kernel_size=(5, 20), stride=(1, 1), padding=(0, 0)),
#             nn.BatchNorm2d(512),
#             nn.ReLU(),

#             nn.Conv2d(512, 512, kernel_size=(5, 20), stride=(1, 1), padding=(0, 0)),
#             nn.BatchNorm2d(512),
#             nn.ReLU(),

#             nn.Conv2d(512, 512, kernel_size=(5, 20), stride=(1, 1), padding=(0, 0)),
#             nn.BatchNorm2d(512),
#             nn.ReLU(),

#             nn.Conv2d(512, 512, kernel_size=(2, 20), stride=(1, 1), padding=(0, 0)),
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#         )

#         # average pooling layer which will convert the output of the convolutional layer to a vector of size 1024 x 1 x 1 x 200
#         self.GlobalAveragePooling = nn.AdaptiveAvgPool2d((1, 200)) 

#         self.fc = nn.Sequential(
#             nn.Linear(512, Image_embedding_size),
#         )


#     def forward(self, x):
#         x = self.convSequence(x)
#         x = self.GlobalAveragePooling(x)

#         x = x.squeeze(2)
#         x = x.permute(0, 2, 1)
#         x = self.fc(x)
#         x = x.permute(0, 2, 1)

#         return x


class EncoderCNN(nn.Module):
    def __init__(self) -> None:
        super(EncoderCNN, self).__init__()
        # input: 30 x 800 
        self.conv_seq = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.AvgPool2d(kernel_size=(2, 4), stride=(2, 4)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=2),

            nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=2),
        )

        self.Linear_seq = nn.Sequential(
            nn.Linear(128 * 7,Image_embedding_size),
            nn.ReLU6(),
        )
    def forward(self, x):
        x = self.conv_seq(x)
        x = x.reshape(x.shape[0], x.shape[3], -1)
        x = self.Linear_seq(x)
        x = x.permute(0, 2, 1)
        return x
    
# cnn = EncoderCNN().to(device)
# test_input = torch.randn(20, 1, 30, 800).to(device)
# test_output = cnn(test_input)
# print(test_output.shape)

# torchsummary.summary(cnn, (1, 30, 800))
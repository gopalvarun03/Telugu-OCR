import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# out look of the model
Image_size = (30, 600) # (height, width)
Image_embedding_size = 512
image_length = 100
Text_embedding_size = 374
Max_Number_of_Words = 350

# Joiner Embedder parameters
Joiner_Input_size = Image_embedding_size #512
Joiner_output_size = Image_embedding_size #512

# LSTM parameters for the RNN
LSTM_Input_size = Joiner_output_size #512
LSTM_hidden_size = LSTM_Input_size #512
LSTM_num_layers = 2
LSTM_output_size = LSTM_hidden_size #512

# reverse Embedding parameters
Reverse_Input_size = LSTM_output_size #512
Reverse_output_size = Text_embedding_size #374

drop_prob = 0.3


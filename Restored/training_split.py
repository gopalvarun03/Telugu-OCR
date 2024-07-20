import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# out look of the model
Image_size = (30, 600) # (height, width)
Image_embedding_size = 364
image_length = 418
Text_embedding_size = 364
Max_Number_of_Words = 350

# Joiner Embedder parameters
Joiner_Input_size = Image_embedding_size #364
Joiner_output_size = Image_embedding_size #364

# LSTM parameters for the RNN
LSTM_Input_size = Joiner_output_size #364
LSTM_hidden_size = LSTM_Input_size #364
LSTM_num_layers = 1
LSTM_output_size = LSTM_hidden_size #364

# reverse Embedding parameters
Reverse_Input_size = LSTM_output_size #364
Reverse_output_size = Text_embedding_size #364

drop_prob = 0.3

acchulu = ['అ', 'ఆ', 'ఇ', 'ఈ', 'ఉ', 'ఊ', 'ఋ', 'ౠ', 'ఌ', 'ౡ', 'ఎ', 'ఏ', 'ఐ', 'ఒ', 'ఓ', 'ఔ', 'అం', 'అః']
hallulu = ['క', 'ఖ', 'గ', 'ఘ', 'ఙ',
           'చ', 'ఛ', 'జ', 'ఝ', 'ఞ',
           'ట', 'ఠ', 'డ', 'ఢ', 'ణ',
           'త', 'థ', 'ద', 'ధ', 'న',
           'ప', 'ఫ', 'బ', 'భ', 'మ',
           'య', 'ర', 'ల', 'వ', 'శ', 'ష', 'స', 'హ', 'ళ', 'క్ష', 'ఱ', 'ఴ', 'ౘ', 'ౙ','ౚ']
vallulu = ['ా', 'ి', 'ీ', 'ు' , 'ూ', 'ృ', 'ౄ', 'ె', 'ే', 'ై', 'ొ', 'ో', 'ౌ', 'ం', 'ః', 'ఁ', 'ఀ', 'ఄ', 'ౕ', 'ౖ', 'ౢ' ]
connector = ['్']
numbers = ['౦', '౧', '౨', '౩', '౪', '౫', '౬', '౭', '౮', '౯']
splcharacters= [' ', '!', '"', '#', '$', '%', '&', "'", '(', ')',
              '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[',
              '\\', ']', '^', '_', '`', '{', '|', '}', '~', '1','2', '3', '4', '5', '6', '7', '8', '9', '0', 'ఽ']
spl = splcharacters + numbers

bases = acchulu + hallulu + spl
vms = vallulu
cms = hallulu

characters = bases+vms+cms+connector

base_mapping = {}
i = 1
for x in bases:
  base_mapping[x] = i
  i+=1

vm_mapping = {}
i = 1
for x in vms:
  vm_mapping[x] = i
  i+=1

cm_mapping = {}
i = 1
for x in cms:
  cm_mapping[x] = i
  i+=1

# creates a list of ductionaries with each dictionary reporesenting a term
def wordsDicts(s):
  List = []
  for i in range(len(s)):
    x = s[i]
    prev = ''
    if i > 0: prev = s[i-1]
    #----------------------------------is it a base term-----------------------
    if((x in acchulu or x in hallulu)  and prev != connector[0]):
      List.append({})
      List[-1]['base'] = x
    #----------------------------if it is a consonant modifier-----------------
    elif x in hallulu and prev == connector[0]:
      if(len(List) == 0):
        print(x)
      if('cm' not in List[-1]): List[-1]['cm'] = []
      List[len(List)-1]['cm'].append(x)

      #---------------------------if it is a vowel modifier--------------------
    elif x in vallulu:
      if(len(List) == 0):
        print(x)

      if('vm' not in List[-1]): List[-1]['vm'] = []
      List[len(List)-1]['vm'].append(x)

      #----------------------------it is a spl character-----------------------
    elif x in spl:
      List.append({})
      List[len(List)-1]['base'] = x
    else:
      continue
  return List

def index_encoding(s):
  List = wordsDicts(s)
  onehot = []
  for i in range(len(List)):
    D = List[i]
    onehotbase=  [0]
    onehotvm1 =  [1]
    onehotvm2 =  [1]
    onehotvm3 =  [1]
    onehotvm4 =  [1]
    onehotcm1 =  [1]
    onehotcm2 =  [1]
    onehotcm3 =  [1]
    onehotcm4 =  [1]


    onehotbase[0] = base_mapping[D['base']]

    it = 1
    if('vm' in D):
      for j in D['vm']:
        if it == 1:
          onehotvm1[0] = vm_mapping[j]+1
        elif it == 2:
          onehotvm2[0] = vm_mapping[j]+1
        elif it == 3:
          onehotvm3[0] = vm_mapping[j]+1
        elif it == 4:
          onehotvm4[0] = vm_mapping[j]+1
        it += 1
    
    it = 1
    if('cm' in D):
      for j in D['cm']:
        if it == 1:
          onehotcm1[0] = cm_mapping[j]+1
        elif it == 2:
          onehotcm2[0] = cm_mapping[j]+1
        elif it == 3:
          onehotcm3[0] = cm_mapping[j]+1
        elif it == 4:
          onehotcm4[0] = cm_mapping[j]+1
        it += 1
    onehoti = onehotbase + onehotvm1 + onehotvm2 + onehotvm3 + onehotvm4 + onehotcm1 + onehotcm2 + onehotcm3 + onehotcm4 #size 112 + 4*21 + 4*40 = 356
    onehot.append(onehoti)
  return onehot

def index_decoder(List):
  x = ""
  for onehoti in List:
    x += bases[onehoti[0]-1]

    if onehoti[5] != 1:
      x += connector[0]
      x += cms[onehoti[5]-2]
    if onehoti[6] != 1:
      x += connector[0]
      x += cms[onehoti[6]-2]
    if onehoti[7] != 1:
      x += connector[0]
      x += cms[onehoti[7]-2]
    if onehoti[8] != 1:
      x += connector[0]
      x += cms[onehoti[8]-2]

    if onehoti[1] != 1:
      x += vms[onehoti[1]-2]
    if onehoti[2] != 1:
      x += vms[onehoti[2]-2]
    if onehoti[3] != 1:
      x += vms[onehoti[3]-2]
    if onehoti[4] != 1:
      x += vms[onehoti[4]-2]
  return x

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class EncoderCNN(nn.Module):
    def __init__(self) -> None:
        super(EncoderCNN, self).__init__()
        # input: 30 x 600  output: 364 x 300
        self.convSequence = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 15), stride=(1, 1), padding=(2, 0)), 
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(5, 15), stride=(1, 1), padding=(2, 0)), 
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, kernel_size=(5, 15), stride=(1, 1), padding=(2, 0)),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(5, 15), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, kernel_size=(5, 15), stride=(1, 1), padding=(2, 0)),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(5, 15), stride=(1, 1), padding=(2, 0)),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(5, 15), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 256, kernel_size=(5, 15), stride=(1, 1), padding=(0, 0)),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(5, 15), stride=(1, 1), padding=(0, 0)),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(5, 15), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 512, kernel_size=(5, 15), stride=(1, 1), padding=(0, 0)),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(5, 15), stride=(1, 1), padding=(0, 0)),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(2, 15), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(512),
        )

        self.GlobalMaxPool = nn.AdaptiveMaxPool2d((1, None))

        self.fc = nn.Linear(512, Image_embedding_size)


    def forward(self, x):
        x = self.convSequence(x)
        x = self.GlobalMaxPool(x)

        x = x.squeeze(2)
        x = x.permute(0, 2, 1)
        x = self.fc(x)
        x = x.permute(0, 2, 1)

        return x
    

class LSTM_Net(nn.Module):
    def __init__(self) -> None:
        super(LSTM_Net, self).__init__()
        # embedding layer sizes
        self.einput_size = Joiner_Input_size #
        self.eoutput_size = Joiner_output_size #200
        # LSTM parameters
        self.embed_size = LSTM_Input_size #200
        self.hidden_size = LSTM_hidden_size #200
        self.num_layers = LSTM_num_layers #1
        # reverse embedding layer sizes
        self.Rinput_size = Reverse_Input_size #200
        self.Routput_size = Reverse_output_size #364

        # dense embedding layers from 50 to 200
        self.embedding1 = nn.Linear(self.einput_size, self.eoutput_size, bias=False)
        
        # LSTM layer
        self.lstm1 = nn.LSTM(input_size = self.embed_size, hidden_size = int(self.embed_size/2) , num_layers = self.num_layers, bidirectional = True, batch_first=True, dropout = drop_prob) #200 to 200
        self.lstm2 = nn.LSTM(input_size = self.embed_size, hidden_size = int(self.embed_size/2), num_layers = self.num_layers, bidirectional = True, batch_first=True, dropout = drop_prob) #200 to 200
        self.lstm3 = nn.LSTM(input_size = self.embed_size, hidden_size = int(self.embed_size/2) , num_layers = self.num_layers, bidirectional = True, batch_first=True, dropout = drop_prob) #200 to 200

        # attention layers for the LSTM
        self.attention_Q = nn.Linear(self.Rinput_size, self.Rinput_size)
        self.attention_K = nn.Linear(self.Rinput_size, self.Rinput_size)
        self.attention_V = nn.Linear(self.Rinput_size, self.Rinput_size)

        # dense layers from 200 to 364
        self.Dense1 = nn.Linear(self.Rinput_size, self.Routput_size, bias=False)
        
        # initialise the weights of the embedding layers
        self.relu = nn.ReLU()
         
    def init_hidden(self, batch_size):
        self.hidden1 = (torch.zeros(2*self.num_layers, batch_size, int(self.embed_size/2)).to(device),
                torch.zeros(2*self.num_layers, batch_size, int(self.embed_size/2)).to(device))

        self.hidden2 = (torch.zeros(2*self.num_layers, batch_size, int(self.embed_size/2)).to(device),
                torch.zeros(2*self.num_layers, batch_size, int(self.embed_size/2)).to(device))

        self.hidden3 = (torch.zeros(2*self.num_layers, batch_size, int(self.embed_size/2)).to(device),
                torch.zeros(2*self.num_layers, batch_size, int(self.embed_size/2)).to(device))
        

    def forward(self, input, New = False):
        if New: # if the input is the image embedding then reset the hidden layers to zeros.
            self.init_hidden(input.shape[0])

        input = self.embedding1(input) # 358 to 17500 
            
        # LSTM layers
        output, self.hidden1 = self.lstm1(input, self.hidden1)
        output, self.hidden2 = self.lstm2(output, self.hidden2)
        output, self.hidden3 = self.lstm3(output, self.hidden3)

        # attention layer
        Q = self.attention_Q(output)
        K = self.attention_K(output)
        V = self.attention_V(output)
        attention = torch.bmm(Q, K.transpose(1, 2))
        attention = F.softmax(attention, dim=2)
        attention = torch.bmm(attention, V)
        
        # dense layer
        attention = F.relu(attention)
        attention = self.Dense1(attention)

        return attention
    
Losses = []

cnn = EncoderCNN().to(device)
network = LSTM_Net().to(device)

# cnn.load_state_dict(torch.load("../Saved_Models/Model_cnn1.pth"))
# network.load_state_dict(torch.load("../Saved_Models/Model_rnn1.pth"))

cnn.train()
network.train()

params = list(network.parameters()) + list(cnn.parameters())
optimizer = optim.Adam(params, lr=1e-3, weight_decay=1e-5)

# gradient clipping
clip = 0.1
torch.nn.utils.clip_grad_norm_(params, clip, norm_type=2, error_if_nonfinite=False)

def read_file_lines(filename):
    lines = []
    try:
        with open(filename, 'r') as file:
            for line in file:
                lines.append(line.strip())  # Remove trailing newline characters
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    return lines

critereon = nn.CTCLoss(blank=0).cuda() if torch.cuda.is_available() else nn.CTCLoss(blank=0)


num_of_epochs = 1000

files = 1

f_output = open("/home/ocr/teluguOCR/output.txt", 'w')

for Epoch in range(1, num_of_epochs + 1):

    Epoch_loss = 0
    start = time.time()
    number_of_images = 0


    for i in range(1, 41):
        print("file: ", i, end = "\r")

        image = torch.load("../Dataset/Full_Image_Tensors/Full_Image_Tensors" + str(i) + ".pt").to(device)
        label = torch.load("../Dataset/Full_Label_Tensors/Full_Label_Tensors" + str(i) + ".pt").to(device)
        target_lengths = torch.load("../Dataset/Full_label_length_tensors/Full_Label_Lengths" + str(i) + ".pt").to(device)

        # randomly selecting 20 images from the batch
        random_indices = torch.randperm(image.shape[0])[:20]

        image = image[random_indices]
        label = label[random_indices]
        target_lengths = target_lengths[random_indices]


        num_of_batches = image.shape[1]
        num_of_images = image.shape[0]
        
        for sub_epochs in range(1, 2):
            # print("sub_epoch: ", sub_epochs)
            # CNN Model

            cnn_output = cnn(image).unsqueeze(1)

            # LSTM Model
            f_out = torch.zeros(num_of_images, Text_embedding_size, image_length).to(device)
            for k in range(image_length):
                f_out[:, :, k] = network(cnn_output[:, :, :, k], k == 0).squeeze(1)
            f_out = f_out.permute(1, 0, 2)

            # softmaxing the output
            f_out[:, :, :112] = F.log_softmax(f_out[:, :, :112], dim=2)
            f_out[:, :, 112:134] = F.log_softmax(f_out[:, :, 112:134], dim=2)
            f_out[:, :, 134:156] = F.log_softmax(f_out[:, :, 134:156], dim=2)
            f_out[:, :, 156:178] = F.log_softmax(f_out[:, :, 156:178], dim=2)
            f_out[:, :, 178:200] = F.log_softmax(f_out[:, :, 178:200], dim=2)
            f_out[:, :, 200:241] = F.log_softmax(f_out[:, :, 200:241], dim=2)
            f_out[:, :, 241:282] = F.log_softmax(f_out[:, :, 241:282], dim=2)
            f_out[:, :, 282:323] = F.log_softmax(f_out[:, :, 282:323], dim=2)
            f_out[:, :, 323:364] = F.log_softmax(f_out[:, :, 323:364], dim=2)
            
            loss = 0

            input_lengths = torch.full(size=(f_out.shape[1],), fill_value=f_out.shape[0], dtype=torch.long).to(device)

            # for base            
            loss += critereon(f_out[:, :, :112], label[:, :, 0], input_lengths, target_lengths)
            # for vm1
            loss += critereon(f_out[:, :, 112:134], label[:, :, 1], input_lengths, target_lengths)
            # for vm2
            loss += critereon(f_out[:, :, 134:156], label[:, :, 2], input_lengths, target_lengths)
            # for vm3
            loss += critereon(f_out[:, :, 156:178], label[:, :, 3], input_lengths, target_lengths)
            # for vm4
            loss += critereon(f_out[:, :, 178:200], label[:, :, 4], input_lengths, target_lengths)
            # for cm1
            loss += critereon(f_out[:, :, 200:241], label[:, :, 5], input_lengths, target_lengths)
            # for cm2
            loss += critereon(f_out[:, :, 241:282], label[:, :, 6], input_lengths, target_lengths)
            # for cm3
            loss += critereon(f_out[:, :, 282:323], label[:, :, 7], input_lengths, target_lengths)
            # for cm4
            loss += critereon(f_out[:, :, 323:364], label[:, :, 8], input_lengths, target_lengths)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            Epoch_loss += loss.item()

            del loss
            del f_out
            del cnn_output
            del input_lengths
            del k

        del image
        del label
        del num_of_images
        del num_of_batches
        del target_lengths

    Losses.append(Epoch_loss)
    print('Epoch: ', Epoch, ' | Loss: ', Epoch_loss, " | Images: ", number_of_images, 'Time: ', time.time() - start)
    f_output.write('Epoch: '+ str(Epoch) + ' | Loss: ' + str(Epoch_loss) + " | Images: " + str(number_of_images) + 'Time: ' + str(time.time() - start))
    del Epoch_loss
    del start
    del number_of_images
    if(Epoch % 100 == 0):
        torch.save(network.state_dict(), "../Saved_Models/Model_rnn" + str(files) + ".pth")
        torch.save(cnn.state_dict(), "../Saved_Models/Model_cnn" + str(files) + ".pth")
        files += 1

f_output.close()

torch.save(torch.stack(Losses), "../Losses.pt")

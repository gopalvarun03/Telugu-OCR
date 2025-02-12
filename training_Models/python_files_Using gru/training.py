from utils import *
from Decoders import DECODER_RNN
from cnn import EncoderCNN
from dataset import TeluguOCRDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from Epoch_Run import *

cnn = EncoderCNN().to(device)
decoder = DECODER_RNN().to(device)

# loss function and optimizer
torch.autograd.set_detect_anomaly(True)
criterion = nn.CTCLoss(blank=0, zero_infinity=True, reduction = 'mean') if torch.cuda.is_available() else nn.CTCLoss(blank=0, zero_infinity=True, reduction = 'mean')

params = list(cnn.parameters()) + list(decoder.parameters())
optimizer = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-5)

clip = 5
torch.nn.utils.clip_grad_norm_(params, clip)

num_of_epochs = 50

Losses = []
val_losses = []

save_num = 1

dataset = TeluguOCRDataset(r"C:/Users/Varun Gopal/Desktop/TeluguOCR_MLProject/KLA_Intern/Dataset/Cropped_Dataset/Images", r"C:/Users/Varun Gopal/Desktop/TeluguOCR_MLProject/KLA_Intern/Dataset/Cropped_Dataset/Labels")

# splitting the dataset into training and validation
torch.manual_seed(0)
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.7, 0.3])

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)

for epoch in range(1, num_of_epochs + 1):
    start_time = time.time()
    # training
    epoch_loss = Epoch_Run(cnn, decoder, train_dataloader, optimizer, criterion, training = True)
    # validation
    val_loss = Epoch_Run(cnn, decoder, val_dataloader, optimizer, criterion, training = False)

    print("Epoch : ", epoch, " | Loss : ", (epoch_loss*64)/len(train_dataset), " | Validation Loss : ", (val_loss*64)/len(val_dataset), " | Time : ", time.time() - start_time)
    Losses.append((epoch_loss*64)/len(train_dataset))
    val_losses.append((val_loss*64)/len(val_dataset))
    if epoch %10 == 0:
        # torch.save(cnn.state_dict(), "/home/ocr/teluguOCR/Models/Best_CNN/GRU1/Model" + str(save_num) + ".pth")
        # torch.save(decoder.state_dict(), "/home/ocr/teluguOCR/Models/Best_RNN/GRU1/Model" + str(save_num) + ".pth")
        save_num += 1
        # saving the losses into a pt file
        # torch.save(torch.tensor(Losses), "/home/ocr/teluguOCR/Losses/Training_GRU1.pt")
        # torch.save(torch.tensor(val_losses), "/home/ocr/teluguOCR/Losses/Validation_GRU1.pt")
        

# Plotting the losses
plt.figure(figsize=(12, 8))
plt.plot(Losses, label = "Training Loss", color = 'blue')
plt.plot(val_losses, label = "Validation Loss", color = 'red')
plt.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.05),
    shadow=True,
    ncol=2
)
plt.title("Losses")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig(r"C:/Users/Varun Gopal/Desktop/TeluguOCR_MLProject/KLA_Intern/Losses_GRU1.png")      
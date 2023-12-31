import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from model import SimpleCNN
import re
from torch.utils.tensorboard import SummaryWriter 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

original_dict = {
    "Ahri": 0,
    "Akali": 1,
    "Alistar": 2,
    "Amumu": 3,
    "Annie": 4,
    "Ashe": 5,
    "Blitzcrank": 6,
    "Braum": 7,
    "Camille": 8,
    "Corki": 9,
    "Darius": 10,
    "Diana": 11,
    "Draven": 12,
    "Evelynn": 13,
    "Ezreal": 14,
    "Fiora": 15,
    "Fizz": 16,
    "Galio": 17,
    "Garen": 18,
    "Gragas": 19,
    "Graves": 20,
    "Janna": 21,
    "Jax": 22,
    "Jhin": 23,
    "Jinx": 24,
    "Katarina": 25,
    "Kennen": 26,
    "Leona": 27,
    "Lulu": 28,
    "Lux": 29,
    "Malphite": 30,
    "Nami": 31,
    "Nasus": 32,
    "Olaf": 33,
    "Orianna": 34,
    "Pantheon": 35,
    "Rakan": 36,
    "Rammus": 37,
    "Rengar": 38,
    "Seraphine": 39,
    "Shyvana": 40,
    "Singed": 41,
    "Sona": 42,
    "Soraka": 43,
    "Teemo": 44,
    "Tristana": 45,
    "Tryndamere": 46,
    "Twisted_Fate": 47,
    "Varus": 48,
    "Vayne": 49,
    "Vi": 50,
    "Xin_Zhao": 51,
    "Yasuo": 52,
    "Wukong": 53,
    "Zed": 54,
    "Ziggs": 55,
    "Dr_Mundo": 56,
    "Ahri": 57,
    "Akali": 58,
    "Alistar": 59,
    "Amumu": 60,
    "Annie": 61,
    "Ashe": 62,
    "Aurelion_Sol": 63,
    "Blitzcrank": 64,
    "Braum": 65,
    "Camille": 66,
    "Corki": 67,
    "Darius": 68,
    "Diana": 69,
    "KaiSa": 70,
    "KhaZix": 71,
    "Jarvan_IV": 72,
    "Master_Yi": 73,
    "Lee_Sin": 74,
    "Miss_Fortune": 75
}

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        self.data_folder = data_folder
        self.transform = transform
        self.image_files = os.listdir(data_folder)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_folder, self.image_files[idx])
        image = Image.open(img_name).convert("RGB")
        # Extract hero name from filename
        img_name = os.path.splitext(img_name)[0]
        img_name = img_name.split("/")[-1]
        element_number = 2  # Change this to the desired element number

        pattern = re.compile(rf'^(.*?)_(\d+)$')
        match = pattern.match(img_name)
        if match:
            hero_name = match.group(1)
        else:
            hero_name = img_name
        
        # Convert hero name to a numerical label (index)
        label = original_dict.get(hero_name, -1)  # Use the original_dict to get the label
        if label == -1:
            raise ValueError(f"Unknown hero name: {hero_name}")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


# Data transformations
transform = transforms.Compose([
    transforms.Resize((42, 42)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
])

# Path to the folder containing images
data_folder = '/home/dattran/datadrive/research/heros_detection/datasets/heroes/train_data'

# Custom dataset and dataloader
dataset = CustomDataset(data_folder, transform=transform)
data_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
# Number of classes (54 in this case)
num_classes = 76

# Initialize the model, loss function, and optimizer
model = SimpleCNN(image_height = 42,image_width = 42, num_classes=num_classes)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Initialize an empty list to store the loss values
losses = []
writer = SummaryWriter('logs') 
num_epochs = 40
for epoch in range(num_epochs):
    total_loss = 0.0
    for i, (images, labels) in enumerate(data_loader):
        # Forward pass
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        if (i + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(data_loader)}], Loss: {loss.item():.4f}')
        
        # Log the loss to TensorBoard every 10 steps
        step = epoch * len(data_loader) + i
        writer.add_scalar('Loss', loss.item(), step)
    
    # Log the total loss at the end of each epoch
    writer.add_scalar('Total Loss', total_loss, epoch)
    
print('Training finished!')

dummy_input = torch.randn(1, 3, 42, 42).to(device)  # Example input tensor with shape (batch_size, channels, height, width)

# Create a SummaryWriter
writer = SummaryWriter('logs')

# Add the model graph to TensorBoard
writer.add_graph(model, dummy_input)

# Close the SummaryWriter when you are done
writer.close()

        

# Display a message indicating that the plot has been saved
print('Training loss plot saved as training_loss_plot.png')

checkpoint_path = f'checkpoint_epoch_{epoch + 1}.pth'
torch.save({
    'epoch': epoch + 1,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': total_loss,
}, checkpoint_path)
print(f'Saved checkpoint at epoch {epoch + 1}: {checkpoint_path}')
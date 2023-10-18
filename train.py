import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from model import SimpleCNN

import matplotlib as plt

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
    "Varus": 47,
    "Vayne": 48,
    "Vi": 49,
    "Wukong": 50,
    "Yasuo": 51,
    "Zed": 52,
    "Ziggs": 53
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
        hero_name = os.path.splitext(self.image_files[idx])[0]  # Extract hero name from filename
        
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
data_loader = DataLoader(dataset, batch_size=2, shuffle=True)
    
# Number of classes (54 in this case)
num_classes = 54

# Initialize the model, loss function, and optimizer
model = SimpleCNN(image_height = 42,image_width = 42, num_classes=num_classes)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Training loop
num_epochs = 40
# Initialize an empty list to store the loss values
losses = []
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
        # Append the current loss to the list
        losses.append(loss.item())
        
print('Training finished!')

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
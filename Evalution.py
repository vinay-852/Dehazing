import torch
import torch.optim as optim
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
import torch.nn as nn
import torchvision.models as models

# Ensure the device is set correctly
device = torch.device("cpu")

# Define HazyDataset
class HazyDataset(Dataset):
    def __init__(self, hazy_files, transform=None):
        self.hazy_files = hazy_files
        self.transform = transform

    def __len__(self):
        return len(self.hazy_files)

    def __getitem__(self, idx):
        hazy_img_path = self.hazy_files[idx]

        hazy_image = Image.open(hazy_img_path).convert('RGB')

        if self.transform:
            hazy_image = self.transform(hazy_image)

        return hazy_image

# Directories for hazy and ground truth images
hazy_dir = "hazing-images"
hazy_files = sorted([os.path.join(hazy_dir, f) for f in os.listdir(hazy_dir)])

# Define transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Create dataset and dataloader
dataset = HazyDataset(hazy_files, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Define DehazingResNet18 model
class DehazingResNet18(nn.Module):
    def __init__(self, num_output_channels=3):
        super(DehazingResNet18, self).__init__()
        
        # Load pre-trained ResNet18
        self.resnet18 = models.resnet18(pretrained=True)
        
        # Remove fully connected layer
        self.resnet18 = nn.Sequential(*list(self.resnet18.children())[:-2])
        
        # Additional convolutional layers for feature extraction
        self.conv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(32, num_output_channels, kernel_size=3, padding=1)
        
        # Batch Normalization layers
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(32)
        
        # Upsampling layers
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Activation function
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # Pass through ResNet18
        x = self.resnet18(x)
        
        # Additional layers for dehazing
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.upsample1(x)
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.upsample2(x)
        
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.upsample3(x)
        
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.upsample4(x)
        
        x = self.conv5(x)
        x = self.upsample5(x)
        
        return x

# Load the model
model = DehazingResNet18().to(device)
model.load_state_dict(torch.load('Dehazing.pth', map_location=device))

# Function to display images
def display_images(hazy_images, dehazed_images):
    fig, axes = plt.subplots(len(hazy_images), 2, figsize=(10, 5 * len(hazy_images)))

    for i in range(len(hazy_images)):
        axes[i, 0].imshow(hazy_images[i])
        axes[i, 0].set_title('Hazy Image')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(dehazed_images[i])
        axes[i, 1].set_title('Dehazed Image')
        axes[i, 1].axis('off')

    plt.show()

# Collect some sample images to display
eval_dataset = dataset
hazy_sample, dehazed_sample = [], []
model.eval()
with torch.no_grad():
    for i in range(5):  # Show 5 samples
        hazy_image = eval_dataset[i].to(device)
        hazy_sample.append(hazy_image.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)  # Denormalize for display

        hazy_image = hazy_image.unsqueeze(0)
        dehazed_image = model(hazy_image).squeeze(0).cpu().permute(1, 2, 0).numpy()
        dehazed_sample.append(dehazed_image * 0.5 + 0.5)  # Denormalize for display

# Display the images
display_images(hazy_sample, dehazed_sample)

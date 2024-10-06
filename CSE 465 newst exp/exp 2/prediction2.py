import torch
import torch.nn as nn
from torchvision.transforms import transforms
import numpy as np
import torch.functional as F  # Removed unused imports
from pathlib import Path
import glob
from PIL import Image
import cv2  # Moved cv2 import here (might not be used)

# Define paths to training and testing data
train_path = '../465/Train'
pred_path = '../465/Pred'

root = Path(train_path)
classes = sorted([j.name.split('/')[-1] for j in root.iterdir()])
print(f"Found classes: {classes}")


class ConvNet(nn.Module):
    def __init__(self,num_classes=43):
        super(ConvNet,self).__init__()
        
        #Output size after convolution filter
        #((w-f+2P)/s) +1
        
        #Input shape= (256,3,150,150)
        
        self.conv1=nn.Conv2d(in_channels=3,out_channels=12,kernel_size=3,stride=1,padding=1)
        #Shape= (256,12,150,150)
        self.bn1=nn.BatchNorm2d(num_features=12)
        #Shape= (256,12,150,150)
        self.relu1=nn.ReLU()
        #Shape= (256,12,150,150)
        
        self.pool=nn.MaxPool2d(kernel_size=2)
        #Reduce the image size be factor 2
        #Shape= (256,12,75,75)
        
        
        self.conv2=nn.Conv2d(in_channels=12,out_channels=20,kernel_size=3,stride=1,padding=1)
        #Shape= (256,20,75,75)
        self.relu2=nn.ReLU()
        #Shape= (256,20,75,75)
        
        
        
        self.conv3=nn.Conv2d(in_channels=20,out_channels=32,kernel_size=3,stride=1,padding=1)
        #Shape= (256,32,75,75)
        self.bn3=nn.BatchNorm2d(num_features=32)
        #Shape= (256,32,75,75)
        self.relu3=nn.ReLU()
        #Shape= (256,32,75,75)
        
        
        self.fc=nn.Linear(in_features=32 * 100 * 100,out_features=num_classes)
        
        
        
        #Feed forward function
        
    def forward(self,input):
        output=self.conv1(input)
        output=self.bn1(output)
        output=self.relu1(output)
            
        output=self.pool(output)
            
        output=self.conv2(output)
        output=self.relu2(output)
            
        output=self.conv3(output)
        output=self.bn3(output)
        output=self.relu3(output)
            
            
            #Above output will be in matrix form, with shape (256,32,75,75)
            
        output=output.view(-1,32*100*100)
            
            
        output=self.fc(output)
            
        return output


# Load pre-trained model weights
checkpoint = torch.load('best_checkpoint.model')
model = ConvNet(num_classes=43)
model.load_state_dict(checkpoint)
model.eval()

# Define transformations for image preprocessing
transformer = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


def prediction(img_path, transformer):
    # Open image and convert to tensor
    image = Image.open(img_path)
    image_tensor = transformer(image).float()

    # Add batch dimension (unsqueeze) for compatibility with the model
    image_tensor = image_tensor.unsqueeze_(0)

    # Use GPU if available
    if torch.cuda.is_available():
        image_tensor = image_tensor.cuda()

    # Removed unused Variable (replaced with direct tensor usage)
    output = model(image_tensor)

    # Get predicted class index and label
    index = output.data.numpy().argmax()
    pred = classes[index]

    return pred


# Get list of image paths in the prediction directory
images_path = glob.glob(pred_path + '/*.png')

# Create dictionary to store predictions for each image
pred_dict = {}
for i in images_path:
    # Extract filename from path and predict class
    filename = i[i.rfind('/') + 1:]
    pred_dict[filename] = prediction(i, transformer)

    # Print prediction with a line gap
    print(f"{filename}: {pred_dict[filename]}")

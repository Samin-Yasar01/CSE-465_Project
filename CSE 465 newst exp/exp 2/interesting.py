# Load libraries
import os
import numpy as np
import torch
import glob
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import torchvision
import pathlib
from sklearn.metrics import precision_score, recall_score, f1_score
import cv2
import matplotlib.pyplot as plt

# Checking for device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Transforms
transformer = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(200, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Dataloader
# Path for training and testing directory
train_path = '../new 465 datasets/Datasets/Train'
test_path = '../new 465 datasets/Datasets/Train445-master'

train_loader = DataLoader(
    torchvision.datasets.ImageFolder(train_path, transform=transformer),
    batch_size=256, shuffle=True
)
test_loader = DataLoader(
    torchvision.datasets.ImageFolder(test_path, transform=transformer),
    batch_size=256, shuffle=True
)

# Categories
root = pathlib.Path(train_path)
classes = sorted([j.name.split('/')[-1] for j in root.iterdir()])
print(classes)

# CNN Network
class ConvNet(nn.Module):
    def __init__(self, num_classes=43):
        super(ConvNet, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=12)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        
        self.conv3 = nn.Conv2d(in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.relu3 = nn.ReLU()
        
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(in_features=32 * 100 * 100, out_features=num_classes)
        
    def forward(self, input):
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu1(output)
        output = self.pool(output)
        
        output = self.conv2(output)
        output = self.relu2(output)
        
        output = self.conv3(output)
        output = self.bn3(output)
        output = self.relu3(output)
        
        output = self.dropout(output)
        
        output = output.view(-1, 32 * 100 * 100)
        output = self.fc(output)
        
        return output

model = ConvNet(num_classes=43).to(device)

# Optimizer and loss function
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
loss_function = nn.CrossEntropyLoss()
num_epochs = 10

# Calculating the size of training and testing images
train_count = len(glob.glob(train_path + '/**/*.png'))
test_count = len(glob.glob(test_path + '/**/*.png'))
print(train_count, test_count)

# Model training and saving best model
best_accuracy = 0.0

for epoch in range(num_epochs):
    # Evaluation and training on training dataset
    model.train()
    train_accuracy = 0.0
    train_loss = 0.0
    
    for i, (images, labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.cpu().data * images.size(0)
        _, prediction = torch.max(outputs.data, 1)
        train_accuracy += int(torch.sum(prediction == labels.data))
    
    train_accuracy = train_accuracy / train_count
    train_loss = train_loss / train_count
    
    # Evaluation on testing dataset
    model.eval()
    
    test_accuracy = 0.0
    true_labels = []
    pred_labels = []
    
    for i, (images, labels) in enumerate(test_loader):
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
        
        outputs = model(images)
        _, prediction = torch.max(outputs.data, 1)
        test_accuracy += int(torch.sum(prediction == labels.data))
        true_labels.extend(labels.cpu().numpy())
        pred_labels.extend(prediction.cpu().numpy())
    
    test_accuracy = test_accuracy / test_count
    precision = precision_score(true_labels, pred_labels, average='macro')
    recall = recall_score(true_labels, pred_labels, average='macro')
    f1 = f1_score(true_labels, pred_labels, average='macro')
    
    print(f'Epoch: {epoch} Train Loss: {train_loss} Train Accuracy: {train_accuracy} Test Accuracy: {test_accuracy}')
    print(f'Precision: {precision} Recall: {recall} F1 Score: {f1}')
    
    # Save the best model
    if test_accuracy > best_accuracy:
        torch.save(model.state_dict(), 'best_checkpoint.model')
        best_accuracy = test_accuracy

# Model Interpretability: Grad-CAM
def get_gradcam(model, image, target_layer):
    model.eval()
    image = image.unsqueeze(0).to(device)
    output = model(image)
    
    pred_class = output.argmax(dim=1)
    pred_score = output.max()
    
    grad = torch.autograd.grad(pred_score, target_layer, retain_graph=True)[0]
    pooled_grad = torch.mean(grad, dim=[0, 2, 3])
    
    target_layer_output = target_layer(image).squeeze()
    for i in range(target_layer_output.size(0)):
        target_layer_output[i, :, :] *= pooled_grad[i]
    
    heatmap = target_layer_output.mean(dim=0).cpu().detach().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    
    return heatmap, pred_class

# Example usage of Grad-CAM
sample_image, _ = next(iter(test_loader))
sample_image = sample_image[0]  # Take the first image from the batch

heatmap, pred_class = get_gradcam(model, sample_image, model.conv3)
heatmap = cv2.resize(heatmap, (sample_image.size(1), sample_image.size(2)))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + sample_image.permute(1, 2, 0).cpu().numpy() * 0.6

plt.imshow(superimposed_img)
plt.title(f'Predicted Class: {classes[pred_class]}')
plt.show()

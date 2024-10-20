#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import csv
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.preprocessing import label_binarize  
from sklearn.model_selection import train_test_split  
from concurrent.futures import ThreadPoolExecutor, as_completed 
from tqdm import tqdm  
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_curve, auc, 
    precision_recall_fscore_support  )
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader  
from torchvision import models, transforms  
import matplotlib.pyplot as plt  

# Define data paths
excel_path = ''
root_path = ''
df = pd.read_csv(excel_path, sep=",", index_col=False)
df.columns = ['barcode', 'type']
image_files = [f for f in os.listdir(root_path) if f.endswith('.jpg')]
len(image_files)

def train_model(model, criterion, optimizer, scheduler, num_epochs=18):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # Initialize a dictionary to store training and validation loss/accuracy/precision/recall/F1 scores
    history = {
        'train': {'loss': [], 'acc': [], 'precision': [], 'recall': [], 'f1': []},
        'val': {'loss': [], 'acc': [], 'precision': [], 'recall': [], 'f1': []}
    }

    for epoch in range(num_epochs):
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()  

            running_loss = 0.0
            running_corrects = 0
            all_preds = []
            all_labels = []

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            # Calculate precision, recall, and F1 score for each class
            precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None, zero_division=0)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
                print(f'Class {i} Precision: {p:.4f} Recall: {r:.4f} F1: {f:.4f}')
            
            history[phase]['loss'].append(epoch_loss)
            history[phase]['acc'].append(epoch_acc)
            history[phase]['precision'].append(precision)
            history[phase]['recall'].append(recall)
            history[phase]['f1'].append(f1)

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    # Save chart data to a CSV file
    with open('.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Phase', 'Class', 'Loss', 'Accuracy', 'Precision', 'Recall', 'F1'])

        for epoch_idx in range(num_epochs):
            for phase in ['train', 'val']:
                epoch_loss = history[phase]['loss'][epoch_idx]
                epoch_acc = history[phase]['acc'][epoch_idx].item()  
                precision = history[phase]['precision'][epoch_idx]
                recall = history[phase]['recall'][epoch_idx]
                f1 = history[phase]['f1'][epoch_idx]
                for class_idx in range(len(precision)):
                    writer.writerow([
                        epoch_idx + 1,
                        phase,
                        class_idx,
                        epoch_loss,
                        epoch_acc,
                        precision[class_idx].item(), 
                        recall[class_idx].item(),    
                        f1[class_idx].item()         
                ])

    model.load_state_dict(best_model_wts)
    return model

class CustomDataset(Dataset):
    def __init__(self, root_path, df, transform=None):
        self.root_path = root_path
        self.df = df
        self.transform = transform
        self.df = self.df[self.df['barcode'].apply(lambda x: os.path.exists(os.path.join(root_path, f"{x}.jpg")))]
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        barcode = self.df.iloc[idx]['barcode']
        label = self.df.iloc[idx]['type'] - 1 
        image_path = os.path.join(self.root_path, f"{barcode}.jpg")
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        return image, label


# Define model transform
transform = transforms.Compose([
    transforms.CenterCrop(896),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


# Create datasets and ensure stratified splitting
train_df, val_df = train_test_split(df, test_size=0.3, shuffle=True, stratify=df['type'], random_state=42)
train_dataset = CustomDataset(root_path, train_df, transform=transform)
val_dataset = CustomDataset(root_path, val_df, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=28, shuffle=True, num_workers=20, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=28, shuffle=True, num_workers=20, pin_memory=True)


# Define and train the model
model_ft = models.resnet152(pretrained=True)

def calculate_size(size, kernel_size=3, stride=1, padding=1):
    return (size + 2 * padding - kernel_size) // stride + 1

input_size = 896
feature_map_size = calculate_size(calculate_size(input_size))

# Number of features in the final fully connected layer of resnet152
num_features = 2048
model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
# Modify the fc layer to match the new feature map size
model_ft.fc = nn.Sequential(
    nn.Linear(num_features, 1024),
    nn.ReLU(inplace=True),
    nn.Linear(1024, 512),
    nn.ReLU(inplace=True),
    nn.Linear(512, 256),
    nn.ReLU(inplace=True),
    nn.Linear(256, 128),
    nn.ReLU(inplace=True),
    nn.Linear(128, 3)
)

device_count = torch.cuda.device_count()
if torch.cuda.is_available() and device_count > 1:
    model_ft = nn.DataParallel(model_ft)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.90)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)
dataloaders = {
    'train': train_dataloader, 
    'val': val_dataloader, 
    'test': test_dataloader}
dataset_sizes = {
    'train': len(train_dataset), 
    'val': len(val_dataset), 
    'test': len(test_dataset)}

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)###

test_loss, test_accuracy = evaluate_model(model_ft, test_dataloader, criterion)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
historytest = {
    'test': {'loss': [], 'acc': []} 
}
historytest['test'] = {'loss': [test_loss], 'acc': [test_accuracy]}
historytest


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pyreadr

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.utils.data.dataloader import default_collate
from torch.nn import CrossEntropyLoss
from torch.cuda.amp import GradScaler, autocast

import torchvision.transforms as transforms
from torchvision import models

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score, roc_curve, auc

from PIL import Image
from umap.umap_ import UMAP

# 1. Definition of unimodal models
class ImageModel(nn.Module):
    def __init__(self, pretrained_model_path):
        super(ImageModel, self).__init__()
        # Load the pre-trained model
        model = torch.load(pretrained_model_path)
        if hasattr(model, 'module'):
            model = model.module
        
        self.features = nn.Sequential(*list(model.children())[:-2])
        # Add a new adaptive average pooling layer to ensure consistency with the output size of the pre-trained model
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Sequential(*list(model.fc.children())[:-1])
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  
        x = self.fc(x)
        return x
    
class TranscriptModel(nn.Module):
    def __init__(self):
        super(TranscriptModel, self).__init__()
        self.fc1 = nn.Linear(500, 30)
        self.ln1 = nn.LayerNorm(30)  
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(30, 30)
        self.ln2 = nn.LayerNorm(30)  
        self.dropout = nn.Dropout(0.5)  
        self.fc3 = nn.Linear(30, 128)

    def forward(self, x):
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
class MethyModel(nn.Module):
    def __init__(self):
        super(MethyModel, self).__init__()
        self.fc1 = nn.Linear(500, 30)
        self.ln1 = nn.LayerNorm(30) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(30, 30)
        self.ln2 = nn.LayerNorm(30)  
        self.dropout = nn.Dropout(0.5)  
        self.fc3 = nn.Linear(30, 128)

    def forward(self, x):
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
class ClinicalModel(nn.Module):
    def __init__(self, num_numeric_features, num_categories_features, embedding_dim):
        super(ClinicalModel, self).__init__()
        self.embedding = nn.Embedding(num_categories_features, embedding_dim)
        self.fc1 = nn.Linear(num_numeric_features + num_categories_features * embedding_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)  
        self.dropout = nn.Dropout(0.3) 

    def forward(self, numeric_inputs, category_inputs):
        category_inputs = category_inputs.long() 
        embedded = self.embedding(category_inputs)
        x = torch.cat([numeric_inputs, embedded.view(embedded.size(0), -1)], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        return x
    

# 2. Definition of the fusion model
class MultiModalLateFusionModel(nn.Module):
    def __init__(self, image_model, transcript_model, methy_model, clinical_model, num_classes, device):
        super(MultiModalLateFusionModel, self).__init__()
        self.image_model = image_model
        self.transcript_model = transcript_model
        self.methy_model = methy_model
        self.clinical_model = clinical_model
        self.num_classes = num_classes
        
        self.weights = [0.5, 0.3, 0.1, 0.1] # Define the weights
        self.fc = nn.Linear(128 * 4, 512).to(device)  # Connect the output of the four models
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 256).to(device)
        self.fc_final = nn.Linear(256, num_classes).to(device)

    def forward(self, image_input, transcript_input, methy_input, numeric_input, category_input):
        # Pass through each model separately
        image_output = self.image_model(image_input)
        transcript_output = self.transcript_model(transcript_input)
        methy_output = self.methy_model(methy_input)
        clinical_output = self.clinical_model(numeric_input, category_input)
        
        # Apply the weights
        image_output = image_output * self.weights[0]
        transcript_output = transcript_output * self.weights[1]
        methy_output = methy_output * self.weights[2]
        clinical_output = clinical_output * self.weights[3]
        
        # Concatenate the outputs from the four models
        combined_output = torch.cat([image_output, transcript_output, methy_output, clinical_output], dim=1)
        
        # Pass through a fully connected layer
        x = self.fc(combined_output)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc_final(x)
        return x

    
num_classes =  

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
image_model = ImageModel('/complete_model.pth').to(device)
transcript_model = TranscriptModel().to(device)
methy_model = MethyModel().to(device)
clinical_model = ClinicalModel(num_numeric_features= , num_categories_features= , embedding_dim=10).to(device) #"Modify according to the specific situation"
model = MultiModalLateFusionModel(image_model, transcript_model, methy_model, clinical_model, num_classes,device).to(device)

# If there are multiple GPUs, use DataParallel for parallelization
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)


# 3. Combined dataset
class CombinedDataset(Dataset):
    def __init__(self, root_path, df, trans, methy, clinical_data, clinical_categories, transform=None, seed=42):
        self.root_path = root_path  
        self.transform =transforms.Compose([
    transforms.CenterCrop(896),  # Center crop to 896x896
    transforms.RandomHorizontalFlip(),  # Random horizontal flip
    transforms.ToTensor(),  
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # Normalize
])
        
        # Get common barcodes for all datasets
        common_barcodes = set(df['barcode']).intersection(set(trans.columns)).intersection(set(methy.columns))
        print(f"Number of common barcodes: {len(common_barcodes)}")
        common_barcodes = list(common_barcodes)
        # Filter the data based on common barcodes
        self.df = df[df['barcode'].isin(common_barcodes)].set_index('barcode').loc[common_barcodes]
        trans = trans.loc[:, common_barcodes]
        methy = methy.loc[:, common_barcodes]
        clinical_data = clinical_data.loc[common_barcodes, :]
        clinical_categories = clinical_categories.loc[common_barcodes, :]

        X_transcript = torch.tensor(trans.values.T, dtype=torch.float32) 
        X_methy = torch.tensor(methy.values.T, dtype=torch.float32) 

        # Perform preliminary dimensionality reduction on transcriptomic data using UMAP and convert to tensor
        y_tensor = torch.tensor(self.df['type'].astype(int).values, dtype=torch.long)
        umap_transcript = UMAP(n_neighbors=15, min_dist=0.3, n_components=500, random_state=seed)
        X_transcript_umap = umap_transcript.fit_transform(X_transcript.numpy(), y=y_tensor.numpy())
        X_transcript_umap_tensor = torch.tensor(X_transcript_umap, dtype=torch.float32)

        # Perform preliminary dimensionality reduction on methylation data using UMAP and convert to tensor
        umap_methy = UMAP(n_neighbors=15, min_dist=0.3, n_components=500, random_state=seed)
        X_methy_umap = umap_methy.fit_transform(X_methy.numpy(), y=y_tensor.numpy())
        X_methy_umap_tensor = torch.tensor(X_methy_umap, dtype=torch.float32)

        # Save the transformed tensors
        self.X_transcript = X_transcript_umap_tensor
        self.X_methy = X_methy_umap_tensor
        self.X_numeric = torch.tensor(clinical_data.values, dtype=torch.float32)
        self.X_category = torch.tensor(clinical_categories.values, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        barcode = self.df.index[idx]
        label = self.df.iloc[idx]['type']
        
        # Load and transform image
        image_path = os.path.join(self.root_path, f"{barcode}.jpg")
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)

        # Get data from the corresponding tensor
        transcript_data = self.X_transcript[idx]
        methy_data = self.X_methy[idx]
        numeric_data = self.X_numeric[idx]
        category_data = self.X_category[idx]

        return image, transcript_data, methy_data, numeric_data, category_data, label


def custom_collate(batch): 
    # Process image, transcriptomic, numeric, categorical, and label data separately
    image = [item[0] for item in batch]
    transcript_data = [item[1] for item in batch]
    methy_data = [item[2] for item in batch]
    numeric_data = [item[3] for item in batch]
    category_data = [item[4] for item in batch]
    label = [item[5] for item in batch]

    # Use the default method to stack image, transcriptomic, numeric, categorical, and label data
    image = default_collate(image)
    transcript_data = default_collate(transcript_data)
    methy_data = default_collate(methy_data)
    numeric_data = default_collate(numeric_data)
    category_data = default_collate(category_data)
    label = default_collate(label)

    return (image, transcript_data, methy_data, numeric_data, category_data), label


# Data loading and processing
trans_path=''
trans = pd.read_csv(trans_path, header=0, index_col=0)

methy_path = ''
methy = pd.read_csv(methy_path, header=0, index_col=0)

clinical_data = ''
clinical_categories = ''
clinical_data = pd.read_excel(clinical_data, header=0, index_col=0)
clinical_categories = pd.read_excel(clinical_categories, header=0, index_col=0)

# Convert clinical_categories to integer encoding and obtain the category mapping
category_mapping = {}
for col in clinical_categories.select_dtypes(include=['object']).columns:
    clinical_categories[col], category_mapping[col] = pd.factorize(clinical_categories[col])
# Handle missing values in clinical_data
clinical_data.replace("'--", pd.NA, inplace=True)
clinical_data = clinical_data.fillna(clinical_data.median(numeric_only=True))
clinical_data = clinical_data.apply(pd.to_numeric, errors='coerce')  # Convert all data to numeric type, filling non-numeric parts with NaN
clinical_data = clinical_data.fillna(0) 

root_path = ''
label_path = ''
df = pd.read_excel(label_path,names=['barcode','type'])
label_mapping = { }

df['type'] = df['type'].replace(label_mapping)
df['barcode'] = df['barcode'].astype(str)

# Create dataset instance
dataset = CombinedDataset(root_path, df, trans, methy, clinical_data, clinical_categories)
# Create data loader using custom collate_fn
data_loader = DataLoader(dataset, batch_size=14, shuffle=True, collate_fn=custom_collate)


# 4. Train and evaluate the model
def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, num_epochs=25, patience=5):
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for (image, transcript_data, methy_data, numeric_data, category_data), label in train_loader:
            image = image.to(device)
            transcript_data = transcript_data.to(device)
            methy_data = methy_data.to(device)
            numeric_data = numeric_data.to(device)
            category_data = category_data.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            outputs = model(image, transcript_data, methy_data, numeric_data, category_data)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {average_loss:.4f}")

        # Validation set evaluation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for (images, transcript_data, methy_data, numeric_data, category_data), labels in val_loader:
                images = images.to(device)
                transcript_data = transcript_data.to(device)
                methy_data = methy_data.to(device)
                numeric_data = numeric_data.to(device)
                category_data = category_data.to(device)
                labels = labels.to(device)

                outputs = model(images, transcript_data, methy_data, numeric_data, category_data)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = correct / total  # Calculate validation accuracy
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")


        # Early stopping mechanism
        if val_loss < best_val_loss and val_accuracy < 0.99:
            best_val_loss = val_loss
            best_val_accuracy = val_accuracy
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break


# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Split the dataset using train_test_split for stratified sampling
labels = [item[-1] for item in dataset]
train_indices, val_indices = train_test_split(
    range(len(labels)), test_size=0.5, stratify=labels, random_state=42
)

# Create training and validation datasets and loaders
train_dataset = torch.utils.data.Subset(dataset, train_indices)
val_dataset = torch.utils.data.Subset(dataset, val_indices)
train_loader = DataLoader(train_dataset, batch_size=14, shuffle=True, collate_fn=custom_collate)
val_loader = DataLoader(val_dataset, batch_size=14, shuffle=False, collate_fn=custom_collate)

# Train the model
train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, num_epochs=25, patience=5)

# Load the best model for evaluation
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
y_true = []
y_pred = []
predicted_probs = []

with torch.no_grad():
    for (images, transcript_data, methy_data, numeric_data, category_data), labels in val_loader:
        # Ensure all data is moved to the same device
        images = images.to(device)
        transcript_data = transcript_data.to(device)
        methy_data = methy_data.to(device)
        numeric_data = numeric_data.to(device)
        category_data = category_data.to(device)
        labels = labels.to(device)

        # Use the model for prediction
        outputs = model(images, transcript_data, methy_data, numeric_data, category_data)
        _, predicted = torch.max(outputs.data, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())
        predicted_probs.extend(outputs.cpu().numpy())

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)
class_accuracies = cm.diagonal() / cm.sum(axis=1)

# Output the accuracy for each class
for i, acc in enumerate(class_accuracies):
    print(f'Class {i} Accuracy: {acc:.4f}')
class_names = [f'Class {i}' for i in range(num_classes)]

# Visualize the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Binarize the labels
y_true_binarized = label_binarize(y_true, classes=range(len(class_names)))

# Compute the P-R curve and AP, mAP
precision = dict()
recall = dict()
average_precision = dict()
num_classes = len(class_names)

for i in range(num_classes):
    probabilities = np.array(predicted_probs)[:, i]
    precision[i], recall[i], _ = precision_recall_curve(y_true_binarized[:, i], probabilities)
    average_precision[i] = average_precision_score(y_true_binarized[:, i], probabilities)

# Plot the P-R curve
plt.figure()
for i in range(num_classes):
    plt.plot(recall[i], precision[i], lw=2, label=f'class {i} (AP = {average_precision[i]:.2f})')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall curve")
plt.legend(loc="best")
plt.show()

# Compute mAP
mAP = np.mean(list(average_precision.values()))
print(f'Mean Average Precision (mAP): {mAP:.4f}')

# Compute and plot the ROC curve
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], np.array(predicted_probs)[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot the ROC curve
plt.figure()
for i in range(num_classes):
    plt.plot(fpr[i], tpr[i], lw=2, label=f'class {i} (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) curve')
plt.legend(loc='best')
plt.show()


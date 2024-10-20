#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#1.UMAP
# Set random seed
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Data loading
data = 
label_path = ''
label = pd.read_excel(label_path, header=None)
label.columns = ['barcode', 'type']  

# Type mapping dictionary
label_mapping = { }
label['type'] = label['type'].replace(label_mapping)
label['barcode'] = label['barcode'].astype(str)

# Get barcodes that exist in the intersection, filter data based on intersection
common_barcodes =data.columns.intersection(label['barcode'])
label = label[label['barcode'].isin(common_barcodes)]
data = data.loc[:, common_barcodes]


X_tensor = torch.tensor(data.values.T, dtype=torch.float32)  
y_tensor = torch.tensor(label['type'].astype(int).values, dtype=torch.long)

# Perform dimensionality reduction using UMAP
umap = UMAP(n_neighbors=15, min_dist=0.3, n_components=500, random_state=seed)
X_umap = umap.fit_transform(X_tensor.numpy(), y=y_tensor.numpy())

selected_dimensions_idx = [3, 18]

# Create a DataFrame to store selected dimensions
umap_selected_df = pd.DataFrame(X_umap[:, selected_dimensions_idx], columns=[f'Dim_{i}' for i in selected_dimensions_idx])

# Color mapping
cmap = ListedColormap(['blue', 'orange', 'green'])
class_labels = sorted(label_mapping.keys())

font = FontProperties()
font.set_family('Arial')
font.set_weight('bold')

# Plot the UMAP results
fig, ax = plt.subplots(figsize=(10, 10))
for cls, color in zip(class_labels, cmap.colors):
    class_mask = (y_tensor.numpy() == label_mapping[cls])
    ax.scatter(umap_selected_df[f'Dim_{selected_dimensions_idx[0]}'][class_mask], 
               umap_selected_df[f'Dim_{selected_dimensions_idx[1]}'][class_mask], 
               c=[color], alpha=0.6, label=cls)

ax.set_xlabel(f'Dim_{selected_dimensions_idx[0]}', fontproperties=font)
ax.set_ylabel(f'Dim_{selected_dimensions_idx[1]}', fontproperties=font)
ax.set_title('UMAP', fontproperties=font)
ax.legend(loc='best')

plt.tight_layout()
plt.show()


#2.CAM

# Load the model
model_path = ''
model_ft = torch.load(model_path)

if isinstance(model_ft, nn.DataParallel):
    model_ft = model_ft.module

# Define a list to store activation outputs
activation_maps = []

# Define a hook function to get activations
def get_activation_hook(module, input, output):
    activation_maps.append(output.detach())

# Register the hook to the output of layer4
hook = model_ft.layer4.register_forward_hook(get_activation_hook)

model_ft = model_ft.cuda()

img_path = ''
image = Image.open(img_path)

# Transform the image
transform = transforms.Compose([
    transforms.Resize((896, 896)),  # Resize the image to match the model input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
input_tensor = input_tensor.cuda()
output = model_ft(input_tensor)

# Get the predicted class
_, pred = torch.max(output, 1)

# Remove the hook
hook.remove()

# Get the feature map from the last convolutional layer
features = activation_maps[0].cpu().numpy()
# Get the weights of the last fully connected layer
params = list(model_ft.fc.parameters())
weight_softmax = params[-2].data.cpu().numpy()

# Compute CAM
def calculate_cam(features, weight_softmax, class_idx):
    b, c, h, w = features.shape
    output_cam = np.zeros((h, w), dtype=np.float32)

    for i, w in enumerate(weight_softmax[class_idx]):
        output_cam += w * features[0, i, :, :]

    output_cam = np.maximum(output_cam, 0)
    output_cam = cv2.resize(output_cam, (896, 896))
    output_cam = output_cam - np.min(output_cam)
    output_cam = output_cam / np.max(output_cam)
    return output_cam

cam = calculate_cam(features, weight_softmax, pred.item())

# Visualize the CAM
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

resized_image = image.resize((896, 896), Image.ANTIALIAS)
resized_image = np.array(resized_image)
resized_image_pil = Image.fromarray(resized_image)  

heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
result = heatmap * 0.3 + resized_image * 0.5
result = Image.fromarray(np.uint8(result))

combined_image = Image.new('RGB', (896 * 2, 896))
combined_image.paste(resized_image_pil, (0, 0)) 
combined_image.paste(result, (896, 0))  

# Save the final combined image
combined_image.save('CAM_combined_result.png')


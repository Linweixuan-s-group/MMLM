#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from PIL import Image, ImageEnhance, ImageStat
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import seaborn as sns
import pandas as pd

# Original image path
root_path = ''
# New folder path
new_folder_path = ''

if not os.path.exists(new_folder_path):
    os.makedirs(new_folder_path)

def calculate_statistics(img):
    stat = ImageStat.Stat(img)
    mean_brightness = np.mean(stat.mean)
    std_contrast = np.mean(stat.stddev)
    hsv_img = img.convert('HSV')
    stat_hsv = ImageStat.Stat(hsv_img)
    mean_saturation = np.mean(stat_hsv.mean[1])
    return mean_brightness, std_contrast, mean_saturation

def calculate_adjustment_factors(orig_value, target_min, target_max):
    if orig_value < target_min:
        return target_min / orig_value
    elif orig_value > target_max:
        return target_max / orig_value
    return 1.0

def process_image(img_filename, root_path, new_folder_path, a, b, saturation_factor, gamma_value):
    original_img_path = os.path.join(root_path, img_filename)
    new_img_path = os.path.join(new_folder_path, img_filename)
    
    with Image.open(original_img_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        orig_brightness, orig_contrast, orig_saturation = calculate_statistics(img)
        
        brightness_factor = calculate_adjustment_factors(orig_brightness, a-1, a+1)
        contrast_factor = calculate_adjustment_factors(orig_contrast, b-1, b+1)
        
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness_factor)
        
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast_factor)
        
        img = img.point(lambda i: 255 * ((i / 255) ** (1 / gamma_value)))
        
        img = np.array(img)
        for c in range(3):  # RGB channels
            offset = 127.5 * (1 - saturation_factor)
            img[..., c] = np.clip(img[..., c] * saturation_factor + offset, 0, 255)
        img = Image.fromarray(np.uint8(img))
        
        adj_brightness, adj_contrast, adj_saturation = calculate_statistics(img)
        img.save(new_img_path)
        
        return orig_brightness, orig_contrast, orig_saturation, adj_brightness, adj_contrast, adj_saturation

brightness_list = []
contrast_list = []
saturation_list = []
adjusted_brightness_list = []
adjusted_contrast_list = []
adjusted_saturation_list = []

a = 158
b = 49
saturation_factor = 0.9
gamma_value = 1.1

# Use ThreadPoolExecutor to process images in parallel
with ThreadPoolExecutor(max_workers=42) as executor:
    futures = []
    for img_filename in os.listdir(root_path):
        if img_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            futures.append(executor.submit(process_image, img_filename, root_path, new_folder_path, a, b, saturation_factor, gamma_value))
    
    # Set the width and update interval of the progress bar
    for future in tqdm(as_completed(futures), desc='Processing Images', unit='file', mininterval=1, maxinterval=5, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'):
        orig_brightness, orig_contrast, orig_saturation, adj_brightness, adj_contrast, adj_saturation = future.result()
        brightness_list.append(orig_brightness)
        contrast_list.append(orig_contrast)
        saturation_list.append(orig_saturation)
        adjusted_brightness_list.append(adj_brightness)
        adjusted_contrast_list.append(adj_contrast)
        adjusted_saturation_list.append(adj_saturation)


brightness_array = np.array(brightness_list)
contrast_array = np.array(contrast_list)
saturation_array = np.array(saturation_list)
adjusted_brightness_array = np.array(adjusted_brightness_list)
adjusted_contrast_array = np.array(adjusted_contrast_list)
adjusted_saturation_array = np.array(adjusted_saturation_list)

# Organize brightness and contrast data into a DataFrame
data = {
    "Original Brightness": brightness_list,
    "Adjusted Brightness": adjusted_brightness_list,
    "Original Contrast": contrast_list,
    "Adjusted Contrast": adjusted_contrast_list,
    "Original saturation": saturation_list,
    "Adjusted saturation": adjusted_saturation_list,
}
df = pd.DataFrame(data)


# Set matplotlib configurations to ensure the correct display of minus signs when saving images
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(21, 7))

# Plot the brightness density curve
plt.subplot(1, 3, 1)
sns.kdeplot(data=df[["Original Brightness"]], fill=True, color="#84CAC0")
plt.title("Brightness Density Curve")
plt.xlabel("Brightness")
plt.ylabel("Density")
plt.legend(["Original"])

# Plot the contrast density curve
plt.subplot(1, 3, 2)
sns.kdeplot(data=df[["Original Contrast"]], fill=True, color="#84CAC0")
plt.title("Contrast Density Curve")
plt.xlabel("Contrast")
plt.ylabel("Density")
plt.legend(["Original"])

# Plot the saturation density curve
plt.subplot(1, 3, 3)
sns.kdeplot(data=df[["Original saturation"]], fill=True, color="#84CAC0")
plt.title("Saturation Density Curve")
plt.xlabel("Saturation")
plt.ylabel("Density")
plt.legend(["Original"])

# Save the image
plt.savefig('original_density_comparison_filled.png', dpi=300, format='tiff', bbox_inches='tight')
plt.show()


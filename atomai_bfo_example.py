import os
import atomai
import atomai.utils.datasets as ds
import numpy as np
import matplotlib.pyplot as plt

from skimage.transform import resize

# ----------------------------
# Paths
# ----------------------------
# dataset_dir = r"C:\Users\jessi\PycharmProjects\AI_ML_Microscopy_Hackathon_2025\atomai_app\data"
bfo_tar_path = r"C:\Users\jessi\PycharmProjects\AI_ML_Microscopy_Hackathon_2025\atomai_app\pretrained\bfo.tar"
# os.makedirs(dataset_dir, exist_ok=True)

# ----------------------------
# Load dataset
# ----------------------------
# dataset = ds.stem_smbfo(download=False, filedir=dataset_dir)
# first_key = list(dataset.keys())[0]
# sample_image = dataset[first_key]["main_image"].astype("float32")
sample_image = plt.imread(r"C:\Users\jessi\PycharmProjects\AI_ML_Microscopy_Hackathon_2025\HAADF-STEM image of a 60nm-thick BiFeO 3 thin film grown on LaAlO .png")
# Normalize to [0,1]
sample_image = (sample_image - sample_image.min()) / (sample_image.max() - sample_image.min())
H, W = sample_image.shape
print("Original image shape:", sample_image.shape)

# ----------------------------
# Center crop (256x256 for BFO model)
# ----------------------------
crop_size = 4096
start_y = (H - crop_size) // 2
start_x = (W - crop_size) // 2
crop = sample_image[start_y:start_y+crop_size, start_x:start_x+crop_size]
print("Cropped image shape:", crop.shape)

# Expand dims for AtomAI (H, W, 1)
crop_input = np.expand_dims(crop, axis=-1)

# ----------------------------
# Load AtomAI BFO model
# ----------------------------
segmentor = atomai.load_model(bfo_tar_path)

# ----------------------------
# Predict segmentation
# ----------------------------
pred_tuple = segmentor.predict(crop_input)
pred_mask = pred_tuple[0]

# ----------------------------
# Handle multi-class output
# ----------------------------
# AtomAI BFO outputs shape: (H, W, n_classes, n_channels)
if pred_mask.ndim == 4:
    # Take argmax across classes
    pred_mask = np.argmax(pred_mask, axis=2)

pred_mask = np.squeeze(pred_mask)
print("Predicted mask shape:", pred_mask.shape)

# ----------------------------
# Visualization
# ----------------------------
fig, axes = plt.subplots(1, 2, figsize=(10,5))
axes[0].imshow(crop, cmap='gray')
axes[0].set_title("Center Crop of STEM Image")
axes[0].axis('off')

axes[1].imshow(pred_mask, cmap='hot')
axes[1].set_title("Predicted Segmentation")
axes[1].axis('off')
plt.show()

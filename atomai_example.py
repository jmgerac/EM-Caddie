import os
import atomai
import atomai.utils.datasets as ds
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from pathlib import Path

# Path to pretrained model tar file, in ./atomai_app/pretrained/G_MD.tar
gmd_tar_path = str(Path("atomai_app/pretrained/G_MD.tar"))

# Dataset folder (where the graphene dataset will be stored)
dataset_dir = str(Path("atomai_app/data"))
os.makedirs(dataset_dir, exist_ok=True)

# Load graphene dataset
dataset = ds.stem_graphene(download=False, filedir=dataset_dir)  # Set download=True if this is the first run
                                                                 # !!! WARNING !!!  1.2 GB

# Get first image from the first image stack as an example
sample_image = dataset[0]["image_data"][0]  # shape (1024, 1024)
sample_image = sample_image.astype("float32") / 255.0  # normalize to 0-1
plt.imsave("graphene_atomai_example_img.png", sample_image, cmap="gray")

# Load the pretrained model
segmentor = atomai.load_model(gmd_tar_path)

# Run inference
pred_mask_tuple = segmentor.predict(sample_image)

# Take the first element and squeeze extra dimensions
pred_mask = np.squeeze(pred_mask_tuple[0])  # shape should become (512, 512)

# Resize predicted mask to match original image (if needed)
pred_mask_resized = resize(pred_mask, sample_image.shape, preserve_range=True)

# Generate side-by-side visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

axes[0].imshow(sample_image, cmap='gray')
axes[0].set_title("Original STEM Frame")
axes[0].axis('off')

axes[1].imshow(pred_mask_resized, cmap='gray')
axes[1].set_title("Predicted Segmentation")
axes[1].axis('off')

plt.show()

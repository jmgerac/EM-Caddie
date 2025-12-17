import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import sys
import TEMUpSamplerNet_app.model as model

sys.modules['model'] = model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Paths ---
model_path = r'TEMUpSamplerNet_app/denoise&bgremoval2x.pth'
img_path   = r'C:\Users\jessi\PycharmProjects\AI_ML_Microscopy_Hackathon_2025\graphene_atomai_example_img.png'

# --- Load image ---
img = Image.open(img_path).convert('L')
transform = transforms.ToTensor()  # scales to [0,1]
img_tensor = transform(img).unsqueeze(0)  # add batch dim

# --- Load model (FULL object) ---
model = torch.load(
    model_path,
    map_location=device,
    weights_only=False
)

model = model.to(device)
model.eval()



# --- Run inference ---
with torch.no_grad():
    img_tensor = img_tensor.to(device)
    output = model(img_tensor)

output_img = output.squeeze().cpu().numpy()

# --- Plot results ---
fig, axes = plt.subplots(1,2,figsize=(10,5))
axes[0].imshow(img, cmap='gray')
axes[0].set_title('Original')
axes[0].axis('off')

axes[1].imshow(output_img, cmap='gray')
axes[1].set_title('Upsampled')
axes[1].axis('off')
plt.show()

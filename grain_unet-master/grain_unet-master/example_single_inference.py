import numpy as np
import cv2
import matplotlib.pyplot as plt

from utility.settings import MODEL_PARAMS
from unet_model.inference.infer import single_inference
from pathlib import Path
print("MODEL_PARAMS:", MODEL_PARAMS)
print("Weights exist:", Path(MODEL_PARAMS).exists())


# -------------------------------
# paths
# -------------------------------
IMAGE_PATH = "al-324-200C_60min_aligned_fov6_0.png"
TMP_PRED_PATH = "_tmp_raw_pred.png"   # still written for compatibility


# -------------------------------
# run inference (TRUE model output)
# -------------------------------
raw_pred = single_inference(
    image_path=IMAGE_PATH,
    output_path=TMP_PRED_PATH,
    model=MODEL_PARAMS
)

print(
    "dtype:", raw_pred.dtype,
    "shape:", raw_pred.shape,
    "min:", raw_pred.min(),
    "max:", raw_pred.max(),
    "mean:", raw_pred.mean(),
    "p99:", np.percentile(raw_pred, 99)
)


# raw_pred is now float32 in [0, 1]
raw_pred = np.squeeze(raw_pred).astype(np.float32)


# -------------------------------
# load input image
# -------------------------------
input_img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
if input_img is None:
    raise RuntimeError(f"Failed to load input image: {IMAGE_PATH}")


# -------------------------------
# diagnostic (optional but useful)
# -------------------------------
print(
    "Prediction stats:",
    "min =", raw_pred.min(),
    "max =", raw_pred.max(),
    "p90/p95/p99 =", np.percentile(raw_pred, [90, 95, 99])
)


# -------------------------------
# boundary-style post-processing
# -------------------------------
# threshold tuned for sparse ridge responses
boundary = raw_pred #> 0.4   # try 0.3–0.6 if needed

boundary = boundary.astype(np.uint8) * 255

# optional: thin / clean boundaries
boundary = cv2.morphologyEx(boundary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
boundary = cv2.Canny(boundary, 50, 150)


# -------------------------------
# display side-by-side
# -------------------------------
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.title("Input Image")
plt.imshow(input_img, cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Predicted Grain Boundaries")
plt.imshow(raw_pred, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()

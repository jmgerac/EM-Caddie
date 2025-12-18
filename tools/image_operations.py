import cv2
from skimage.transform import resize
import numpy as np
import torch
import tempfile
import os
from unet_model.inference.infer import single_inference
from utility.settings import MODEL_PARAMS


def hex_to_bgr(hex_color):
    """
    Convert hex color string to BGR tuple for OpenCV.

    :param hex_color: Hex color string (e.g., "#FFFFFF" or "FFFFFF")
    :return: BGR tuple (B, G, R)
    """
    # Remove '#' if present
    hex_color = hex_color.lstrip('#')
    # Convert to RGB
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    # Return as BGR for OpenCV
    return (b, g, r)




def edge_detect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 180)
    return edges

def blur(img):
    return cv2.GaussianBlur(img, (11, 11), 0)

def invert(img):
    return 255 - img

def fft(img):
    # Ensure grayscale
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = img.astype(np.float32)

    # FFT
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = np.log1p(np.abs(fshift))

    # Safe normalization
    mag_min = magnitude.min()
    mag_max = magnitude.max()

    if mag_max > mag_min:
        magnitude = (magnitude - mag_min) / (mag_max - mag_min)
    else:
        magnitude = np.zeros_like(magnitude)

    # Convert to uint8 [0,255]
    magnitude_uint8 = (magnitude * 255).astype(np.uint8)

    return magnitude_uint8




def atomai_segment(img, segmentor):
    # Convert to grayscale if BGR
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Normalize to 0-1 float32
    norm_img = gray.astype("float32") / 255.0

    # Run model
    pred_mask_tuple = segmentor.predict(norm_img)
    pred_mask = np.squeeze(pred_mask_tuple[0])

    # Resize mask to match original image
    pred_mask_resized = resize(pred_mask, gray.shape, preserve_range=True)

    # Convert to uint8 0-255 for display
    pred_mask_uint8 = (pred_mask_resized * 255).clip(0, 255).astype("uint8")
    return pred_mask_uint8

def grain_unet_segment(img, model=None, device=None):
    """
    Grain boundary inference using official single_inference pipeline.
    Mirrors the working standalone example.
    """

    # --- write input image to temp file ---
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_in:
        input_path = tmp_in.name
        cv2.imwrite(input_path, img)

    # --- create temp output path ---
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_out:
        output_path = tmp_out.name

    # --- run official inference ---
    raw_pred = single_inference(
        image_path=input_path,
        output_path=output_path,
        model=MODEL_PARAMS
    )

    # cleanup temp files
    os.remove(input_path)
    os.remove(output_path)

    # --- normalize for display ---
    raw_pred = np.squeeze(raw_pred).astype(np.float32)
    return (raw_pred * 255).clip(0, 255).astype(np.uint8)





def super_res(img, super_res_model, device):

    # Force grayscale conversion
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.ndim != 2:
        raise ValueError(f"Unexpected image shape: {img.shape}")

    # Normalize to 0-1 float32
    img_input = img.astype(np.float32) / 255.0

    # Add batch & channel dims for model: 1 x 1 x H x W
    img_tensor = torch.from_numpy(img_input).unsqueeze(0).unsqueeze(0).to(device)

    # Debug: print shape
    print(f"Input tensor shape: {img_tensor.shape}")

    with torch.no_grad():
        output = super_res_model(img_tensor)
        output_img = output.squeeze().cpu().numpy()

    output_img = (output_img * 255).clip(0, 255).astype(np.uint8)
    return output_img


def add_scale_bar(img, scale_bar_length, units="nm",
                  input_mode="total_length", pixel_to_scale=None, total_image_length=None,
                  position="bottom_right", x_offset=None, y_offset=None,
                  font_scale_factor=1.0, show_background=True, bar_color="#FFFFFF", font_color="#FFFFFF"):
    """
    Add a scale bar to the image with customizable settings.

    :param img: Input image (BGR or grayscale)
    :param scale_bar_length: Desired length of the scale bar in the specified units
    :param units: Unit string (e.g., "nm", "μm", "mm", "px")
    :param input_mode: "pixel_to_scale" or "total_length"
    :param pixel_to_scale: If input_mode is "pixel_to_scale", this is the length per pixel
    :param total_image_length: If input_mode is "total_length", this is the total image length
    :param position: "bottom_right", "bottom_left", "top_right", "top_left", or "custom"
    :param x_offset: Custom X position (0-1, where 0 is left, 1 is right). Used if position="custom"
    :param y_offset: Custom Y position (0-1, where 0 is top, 1 is bottom). Used if position="custom"
    :param font_scale_factor: Multiplier for font size (default 1.0)
    :param show_background: If True, show solid black background box (default True)
    :param bar_color: Hex color string for the scale bar (default "#FFFFFF" = white)
    :param font_color: Hex color string for the font text (default "#FFFFFF" = white)
    :return: Image with scale bar added
    """
    # Make a copy to avoid modifying the original
    result = img.copy()

    # Get image dimensions
    if len(result.shape) == 3:
        height, width = result.shape[:2]
        is_color = True
    else:
        height, width = result.shape
        is_color = False
        # Convert grayscale to BGR for consistent drawing
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    # Calculate pixel length of the scale bar
    if input_mode == "pixel_to_scale":
        # pixel_to_scale is length per pixel (e.g., 0.1 nm/pixel)
        # scale_bar_length is the desired length in units
        # scale_bar_pixels = scale_bar_length / pixel_to_scale
        scale_bar_pixels = int(scale_bar_length / pixel_to_scale)
    elif input_mode == "total_length":
        # total_image_length is the total image size in units
        # scale_bar_length is the desired scale bar length in units
        # scale_bar_pixels = (scale_bar_length / total_image_length) * image_width
        scale_bar_pixels = int((scale_bar_length / total_image_length) * width)
    else:
        # Default fallback
        scale_bar_pixels = int(width * 0.1)  # 10% of image width

    # Ensure scale bar doesn't exceed image width
    scale_bar_pixels = min(scale_bar_pixels, width - 20)
    scale_bar_pixels = max(scale_bar_pixels, 10)  # Minimum 10 pixels

    # Scale bar dimensions
    bar_height = max(3, int(height * 0.01))  # 1% of image height, minimum 3 pixels
    bar_thickness = max(2, int(bar_height * 0.3))  # Thickness of the bar

    # Calculate position
    margin = int(min(width, height) * 0.02)  # 2% margin from edges

    if position == "bottom_right":
        bar_x_start = width - scale_bar_pixels - margin
        bar_x_end = width - margin
        bar_y = height - margin
    elif position == "bottom_left":
        bar_x_start = margin
        bar_x_end = margin + scale_bar_pixels
        bar_y = height - margin
    elif position == "top_right":
        bar_x_start = width - scale_bar_pixels - margin
        bar_x_end = width - margin
        bar_y = margin + bar_height + 20
    elif position == "top_left":
        bar_x_start = margin
        bar_x_end = margin + scale_bar_pixels
        bar_y = margin + bar_height + 20
    elif position == "custom" and x_offset is not None and y_offset is not None:
        # x_offset and y_offset are 0-1, where 0 is left/top, 1 is right/bottom
        # Allow movement across the entire image
        # Calculate available space (accounting for scale bar width and text height)
        text_height_estimate = int(height * 0.03)  # Estimate for text height
        available_width = width - scale_bar_pixels
        available_height = height - bar_thickness - text_height_estimate

        # Position scale bar based on offsets (0 = left/top, 1 = right/bottom)
        bar_x_start = int(x_offset * available_width)
        bar_x_end = bar_x_start + scale_bar_pixels
        # For y: 0 = top, 1 = bottom, so invert the calculation
        bar_y = int((1 - y_offset) * available_height) + bar_thickness
    else:
        # Default to bottom right
        bar_x_start = width - scale_bar_pixels - margin
        bar_x_end = width - margin
        bar_y = height - margin

    # Ensure bar stays within image bounds
    bar_x_start = max(0, min(bar_x_start, width - scale_bar_pixels))
    bar_x_end = min(width, bar_x_start + scale_bar_pixels)
    bar_y = max(bar_thickness, min(bar_y, height))

    # Calculate text dimensions first to properly size the background box
    label = f"{scale_bar_length} {units}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    base_font_scale = max(0.3, width / 2500.0)  # Base font scale based on image size
    font_scale = base_font_scale * font_scale_factor
    font_thickness = max(1, int(font_scale * 2))

    # Get text size for centering and background sizing
    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
    text_x = bar_x_start + (scale_bar_pixels - text_width) // 2
    text_y = bar_y - bar_thickness - 5

    # Calculate background box dimensions to encompass both bar and text
    # Background should cover: scale bar + text above it
    # All elements move together as a unit
    bg_padding = 5
    bg_left = min(bar_x_start - bg_padding, text_x - bg_padding)
    bg_right = max(bar_x_end + bg_padding, text_x + text_width + bg_padding)
    bg_top = max(0, text_y - text_height - bg_padding)
    bg_bottom = min(height, bar_y + bg_padding)

    # Ensure background box stays within image bounds
    bg_left = max(0, bg_left)
    bg_right = min(width, bg_right)
    bg_top = max(0, bg_top)
    bg_bottom = min(height, bg_bottom)

    # Convert hex colors to BGR
    bar_color_bgr = hex_to_bgr(bar_color)
    font_color_bgr = hex_to_bgr(font_color)

    # Draw background rectangle if enabled
    if show_background:
        cv2.rectangle(result,
                      (bg_left, bg_top),
                      (bg_right, bg_bottom),
                      (0, 0, 0), -1)

    # Draw scale bar with specified color (moves with background)
    cv2.rectangle(result,
                  (bar_x_start, bar_y - bar_thickness),
                  (bar_x_end, bar_y),
                  bar_color_bgr, -1)

    # Draw text with outline for better visibility (moves with scale bar)
    # Use black outline if background is shown, otherwise use contrasting outline
    if show_background:
        outline_color = (0, 0, 0)  # Black outline when background is shown
    else:
        # Use a contrasting outline (inverse of font color)
        outline_color = tuple(255 - c for c in font_color_bgr)

    cv2.putText(result, label, (text_x, text_y), font, font_scale, outline_color, font_thickness + 2, cv2.LINE_AA)
    cv2.putText(result, label, (text_x, text_y), font, font_scale, font_color_bgr, font_thickness, cv2.LINE_AA)

    # Convert back to grayscale if original was grayscale
    if not is_color:
        result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    return result
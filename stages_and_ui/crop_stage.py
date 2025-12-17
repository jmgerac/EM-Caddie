"""
Crop Stage Module
Handles image cropping functionality
"""
import streamlit as st
import numpy as np
import cv2
from PIL import Image
from streamlit_cropper import st_cropper


def image_cropper():
    """
    Stage for cropping the uploaded image.
    Shows original dimensions and allows user to crop with real-time dimension display.
    """
    st.markdown("<h2 style='text-align:center;'>Crop Your Image</h2>", unsafe_allow_html=True)
    
    img = st.session_state.get("original_img")
    if img is None:
        st.error("No image loaded. Please go back and upload an image.")
        if st.button("Back to Upload"):
            st.session_state.stage = 2
            st.rerun()
        return
    
    # Convert OpenCV image (BGR) to PIL Image (RGB)
    if len(img.shape) == 3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    # Get original dimensions
    original_width, original_height = pil_img.size
    
    # Display original dimensions
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info(f"**Original Image Dimensions:** {original_width} × {original_height} pixels")
    
    # Cropper settings
    with st.expander("Cropper Settings", expanded=False):
        realtime_update = st.checkbox("Update in Real Time", value=True, key="crop_realtime")
        box_color = st.color_picker("Box Color", value='#FF0000', key="crop_box_color")
        aspect_ratio = st.selectbox(
            "Aspect Ratio",
            ["Free", "1:1", "4:3", "16:9", "3:2"],
            key="crop_aspect_ratio"
        )
        
        # Convert aspect ratio string to tuple or None
        aspect_dict = {
            "Free": None,
            "1:1": (1, 1),
            "4:3": (4, 3),
            "16:9": (16, 9),
            "3:2": (3, 2)
        }
        aspect_ratio_value = aspect_dict[aspect_ratio]
    
    # Initialize or get crop dimensions from session state
    if "crop_width" not in st.session_state:
        st.session_state.crop_width = min(original_width, 500)
    if "crop_height" not in st.session_state:
        st.session_state.crop_height = min(original_height, 500)
    
    # Display cropper
    st.markdown("### Select the area to crop:")
    cropped_img = st_cropper(
        pil_img,
        realtime_update=realtime_update,
        box_color=box_color,
        aspect_ratio=aspect_ratio_value,
        key="image_cropper"
    )
    
    # Get cropped dimensions and handle manual input
    cropped_img = _handle_crop_dimensions(
        cropped_img, pil_img, original_width, original_height
    )
    
    # Display the cropped image and action buttons
    _display_crop_preview_and_actions(cropped_img, pil_img)


def _handle_crop_dimensions(cropped_img, pil_img, original_width, original_height):
    """Handle crop dimension inputs and manual adjustments"""
    use_cropper_result = True
    
    if cropped_img:
        cropped_width, cropped_height = cropped_img.size
        # Only update session state if user hasn't manually set dimensions
        if "manual_crop_set" not in st.session_state or not st.session_state.manual_crop_set:
            st.session_state.crop_width = cropped_width
            st.session_state.crop_height = cropped_height
    
    # Display editable dimension inputs
    st.markdown("### 📐 Cropping Box Dimensions (Click to Edit):")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        input_width = st.number_input(
            "Width (px):",
            min_value=1,
            max_value=original_width,
            value=int(st.session_state.crop_width),
            step=1,
            key="crop_width_input",
            help="Click to input the desired width in pixels"
        )
    
    with col2:
        input_height = st.number_input(
            "Height (px):",
            min_value=1,
            max_value=original_height,
            value=int(st.session_state.crop_height),
            step=1,
            key="crop_height_input",
            help="Click to input the desired height in pixels"
        )
    
    with col3:
        total_pixels = input_width * input_height
        st.metric("Total Pixels", f"{total_pixels:,}")
    
    if st.session_state.get("manual_crop_set", False):
        if st.button("🔄 Use Cropper Selection", use_container_width=True, key="reset_manual_crop"):
            st.session_state.manual_crop_set = False
            st.rerun()
    
    # Check if dimensions were manually changed
    dimensions_changed = (
        input_width != st.session_state.crop_width or 
        input_height != st.session_state.crop_height
    )
    
    if dimensions_changed:
        st.session_state.crop_width = input_width
        st.session_state.crop_height = input_height
        st.session_state.manual_crop_set = True
        
        # Crop from bottom-right corner
        cropped_img = _crop_from_bottom_right(
            pil_img, original_width, original_height, input_width, input_height
        )
        use_cropper_result = False
    elif cropped_img and ("manual_crop_set" not in st.session_state or not st.session_state.manual_crop_set):
        use_cropper_result = True
    else:
        # Use manual dimensions if they were previously set
        cropped_img = _crop_from_bottom_right(
            pil_img, original_width, original_height, 
            st.session_state.crop_width, st.session_state.crop_height
        )
        use_cropper_result = False
    
    return cropped_img


def _crop_from_bottom_right(pil_img, original_width, original_height, width, height):
    """Crop image from bottom-right corner with specified dimensions"""
    left = max(0, original_width - width)
    top = max(0, original_height - height)
    right = min(left + width, original_width)
    bottom = min(top + height, original_height)
    
    # Adjust if we're at the edge
    if right - left < width:
        left = max(0, right - width)
    if bottom - top < height:
        top = max(0, bottom - top)
    
    cropped_img = pil_img.crop((left, top, right, bottom))
    st.info(f"✅ Cropped from bottom-right corner: ({left}, {top}) to ({right}, {bottom})")
    
    return cropped_img


def _display_crop_preview_and_actions(cropped_img, pil_img):
    """Display cropped image preview and action buttons"""
    if cropped_img:
        # Get actual dimensions of cropped image
        final_width, final_height = cropped_img.size
        
        # Show preview
        st.markdown("### Preview:")
        st.image(
            cropped_img, 
            caption=f"Cropped Image ({final_width} × {final_height} pixels)", 
            use_container_width=True
        )
        
        # Action buttons
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("← Back to Upload", use_container_width=True):
                st.session_state.stage = 2
                st.rerun()
        
        with col2:
            if st.button("Use Original Image", use_container_width=True):
                st.session_state.working_img = st.session_state.original_img.copy()
                st.session_state.stage = 4
                st.rerun()
        
        with col3:
            if st.button("Use Cropped Image →", use_container_width=True, type="primary"):
                # Convert PIL image back to OpenCV format
                cropped_array = np.array(cropped_img)
                if len(cropped_array.shape) == 3:
                    cropped_cv = cv2.cvtColor(cropped_array, cv2.COLOR_RGB2BGR)
                else:
                    cropped_cv = cropped_array
                
                # Update both original and working images
                st.session_state.original_img = cropped_cv.copy()
                st.session_state.working_img = cropped_cv.copy()
                st.session_state.stage = 4
                st.rerun()
    else:
        st.warning("Please select a cropping area.")
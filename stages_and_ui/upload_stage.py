"""
Upload Stage Module
Handles initial image upload and scale/cropping functionality
"""
import streamlit as st
import numpy as np
import cv2
from PIL import Image
from streamlit_cropper import st_cropper


def initial_upload():
    """Stage for initial image upload and scale/cropping options"""
    st.markdown("<h2 style='text-align:center;'>EM Caddie</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Upload an image to get started</p>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "", 
        type=["png", "jpg", "jpeg"], 
        key="startup_upload",
        label_visibility="collapsed"
    )
    
    if uploaded_file:
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        cv2_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.session_state.original_img = cv2_img
        
        # Initialize crop mode if not set
        if "crop_mode" not in st.session_state:
            st.session_state.crop_mode = "full"  # "full", "draw", or "coordinates"
        
        # Convert to RGB for display
        if len(cv2_img.shape) == 3:
            img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_GRAY2RGB)
        
        height, width = img_rgb.shape[:2]
        
        # Convert to PIL for cropping operations
        pil_img = Image.fromarray(img_rgb)
        original_width, original_height = pil_img.size
        
        # Initialize cropper settings if not set (needed for draw mode)
        if "crop_realtime_draw" not in st.session_state:
            st.session_state.crop_realtime_draw = True
        if "crop_box_color_draw" not in st.session_state:
            st.session_state.crop_box_color_draw = '#FF0000'
        if "crop_aspect_ratio_value" not in st.session_state:
            st.session_state.crop_aspect_ratio_value = None
        
        # Create two-column layout
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            if st.session_state.crop_mode == "draw":
                # For draw mode, show the cropper in the left column
                st.markdown("### Editable Cropping Area")
                
                # Get settings from session state (will be set in right column)
                realtime_update = st.session_state.get("crop_realtime_draw", True)
                box_color = st.session_state.get("crop_box_color_draw", '#FF0000')
                aspect_ratio_value = st.session_state.get("crop_aspect_ratio_value", None)
                
                # Show cropper
                cropped_img = st_cropper(
                    pil_img,
                    realtime_update=realtime_update,
                    box_color=box_color,
                    aspect_ratio=aspect_ratio_value,
                    key="image_cropper_main"
                )
                
                # Store cropped image in session state
                if cropped_img:
                    st.session_state.cropped_img_draw = cropped_img
                    crop_w, crop_h = cropped_img.size
                else:
                    st.session_state.cropped_img_draw = None
                    crop_w, crop_h = 0, 0
                
                # Display dimensions below cropper
                st.markdown("---")
                col_dim1, col_dim2 = st.columns(2)
                with col_dim1:
                    st.metric("Original Image Size", f"{original_width} × {original_height} pixels")
                with col_dim2:
                    if cropped_img:
                        st.metric("Cropping Area Size", f"{crop_w} × {crop_h} pixels")
                    else:
                        st.metric("Cropping Area Size", "Not selected")
                        
            elif st.session_state.crop_mode == "coordinates":
                # For coordinates mode, show image with bounding box overlay
                st.markdown("### Image Preview with Bounding Box")
                
                # Get crop coordinates from session state (bottom-left origin system)
                origin_x = st.session_state.get("crop_origin_x", 0)
                origin_y_bottom = st.session_state.get("crop_origin_y", 0)  # y coordinate from bottom
                crop_w = st.session_state.get("crop_width", min(original_width, 500))
                crop_h = st.session_state.get("crop_height", min(original_height, 500))
                
                # Convert from bottom-left origin to top-left origin for image operations
                # In bottom-left system: origin_y_bottom = 0 is at bottom, increases upward
                # In top-left system: top_y = original_height - origin_y_bottom - crop_h
                left = origin_x
                top = original_height - origin_y_bottom - crop_h
                right = min(left + crop_w, original_width)
                bottom = min(top + crop_h, original_height)
                
                # Ensure coordinates are within bounds
                top = max(0, top)
                left = max(0, min(left, original_width - 1))
                right = min(original_width, max(left + 1, right))
                bottom = min(original_height, max(top + 1, bottom))
                
                # Draw bounding box on image
                img_with_box = img_rgb.copy()
                # Convert to numpy array for drawing (RGB format)
                img_array = np.array(img_with_box)
                # Draw rectangle in red (RGB: 255, 0, 0) with thickness of 3 pixels
                # Note: cv2.rectangle expects BGR, but we're working with RGB, so we use (0, 0, 255) for red
                # Actually, let's use PIL ImageDraw instead to avoid color conversion issues
                from PIL import ImageDraw
                img_with_box_pil = Image.fromarray(img_array)
                draw = ImageDraw.Draw(img_with_box_pil)
                # Draw rectangle outline (red color in RGB)
                draw.rectangle([(int(left), int(top)), (int(right), int(bottom))], outline=(255, 0, 0), width=3)
                
                st.image(img_with_box_pil, caption=f"Original Image ({original_width} × {original_height} pixels) with Crop Box", use_container_width=True)
                
            else:
                # For full mode, show regular preview
                st.markdown("### Image Preview")
                st.image(img_rgb, caption=f"Original Image ({original_width} × {original_height} pixels)", use_container_width=True)
        
        with col_right:
            st.markdown("### Scale & Crop Options")
            
            # Crop mode selection
            crop_mode = st.radio(
                "Select Option:",
                ["Use Full Sized Image", "Draw Cropping", "Input Cropping Coordinates"],
                index={"full": 0, "draw": 1, "coordinates": 2}.get(st.session_state.crop_mode, 0),
                key="crop_mode_radio"
            )
            st.session_state.crop_mode = (
                "full" if crop_mode == "Use Full Sized Image" 
                else "draw" if crop_mode == "Draw Cropping" 
                else "coordinates"
            )
            
            st.markdown("---")
            
            # Initialize session state for crop coordinates
            if "crop_origin_x" not in st.session_state:
                st.session_state.crop_origin_x = 0
            if "crop_origin_y" not in st.session_state:
                st.session_state.crop_origin_y = 0
            if "crop_width" not in st.session_state:
                st.session_state.crop_width = min(original_width, 500)
            if "crop_height" not in st.session_state:
                st.session_state.crop_height = min(original_height, 500)
            
            # Handle different crop modes
            if st.session_state.crop_mode == "full":
                st.info("Using the full-sized image without cropping.")
                preview_img = pil_img
                
            elif st.session_state.crop_mode == "draw":
                st.markdown("**Draw Cropping Area:**")
                
                # Cropper settings
                with st.expander("Cropper Settings", expanded=False):
                    realtime_update = st.checkbox("Update in Real Time", value=True, key="crop_realtime")
                    box_color = st.color_picker("Box Color", value='#FF0000', key="crop_box_color")
                    aspect_ratio = st.selectbox(
                        "Aspect Ratio",
                        ["Free", "1:1", "4:3", "16:9", "3:2"],
                        key="crop_aspect_ratio"
                    )
                    
                    aspect_dict = {
                        "Free": None,
                        "1:1": (1, 1),
                        "4:3": (4, 3),
                        "16:9": (16, 9),
                        "3:2": (3, 2)
                    }
                    aspect_ratio_value = aspect_dict[aspect_ratio]
                    st.session_state.crop_aspect_ratio_value = aspect_ratio_value
                    st.session_state.crop_realtime_draw = realtime_update
                    st.session_state.crop_box_color_draw = box_color
                
                # Get cropped image from session state (set by main area cropper)
                # This will be updated when the cropper is used
                preview_img = st.session_state.get("cropped_img_draw", None)
                if preview_img is None:
                    st.info("Use the cropper in the main area to draw your crop selection.")
                
            else:  # coordinates mode
                st.markdown("**Input Cropping Coordinates:**")
                st.caption("Note: Origin (0,0) is at the bottom-left of the image.")
                
                st.markdown("**Origin Coordinates (bottom-left origin):**")
                col_ox, col_oy = st.columns(2)
                with col_ox:
                    origin_x = st.number_input(
                        "Origin X:",
                        min_value=0,
                        max_value=original_width - 1,
                        value=int(st.session_state.crop_origin_x),
                        step=1,
                        key="crop_origin_x_input",
                        help="X coordinate from left edge"
                    )
                    st.session_state.crop_origin_x = origin_x
                
                with col_oy:
                    origin_y_bottom = st.number_input(
                        "Origin Y (from bottom):",
                        min_value=0,
                        max_value=original_height - 1,
                        value=int(st.session_state.crop_origin_y),
                        step=1,
                        key="crop_origin_y_input",
                        help="Y coordinate from bottom edge (0 = bottom of image)"
                    )
                    st.session_state.crop_origin_y = origin_y_bottom
                
                st.markdown("**Crop Dimensions:**")
                col_w, col_h = st.columns(2)
                with col_w:
                    crop_w = st.number_input(
                        "Width (px):",
                        min_value=1,
                        max_value=original_width - origin_x,
                        value=int(st.session_state.crop_width),
                        step=1,
                        key="crop_width_input"
                    )
                    st.session_state.crop_width = crop_w
                
                with col_h:
                    # Max height depends on origin_y from bottom
                    max_height_from_bottom = original_height - origin_y_bottom
                    crop_h = st.number_input(
                        "Height (px):",
                        min_value=1,
                        max_value=max_height_from_bottom,
                        value=int(st.session_state.crop_height),
                        step=1,
                        key="crop_height_input"
                    )
                    st.session_state.crop_height = crop_h
                
                # Convert from bottom-left origin to top-left origin for cropping
                # top_y = original_height - origin_y_bottom - crop_h
                top = original_height - origin_y_bottom - crop_h
                left = origin_x
                right = min(left + crop_w, original_width)
                bottom = min(top + crop_h, original_height)
                
                # Ensure coordinates are within bounds
                top = max(0, top)
                left = max(0, min(left, original_width - 1))
                right = min(original_width, max(left + 1, right))
                bottom = min(original_height, max(top + 1, bottom))
                
                # Create preview
                preview_img = pil_img.crop((left, top, right, bottom))
            
            # Show preview above buttons
            if preview_img:
                st.markdown("---")
                st.markdown("**Preview:**")
                prev_width, prev_height = preview_img.size
                st.image(preview_img, caption=f"Preview ({prev_width} × {prev_height} pixels)", use_container_width=True)
            
            st.markdown("---")
            
            # Action buttons
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("← Back to Upload", use_container_width=True):
                    st.session_state.stage = 2
                    st.rerun()
            
            with col_btn2:
                if st.button("Use Image →", use_container_width=True, type="primary", key="use_image_btn"):
                    if st.session_state.crop_mode == "full":
                        st.session_state.working_img = cv2_img.copy()
                        st.session_state.stage = 4
                        st.rerun()
                    elif st.session_state.crop_mode == "draw":
                        # Get cropped image from session state
                        cropped_img_draw = st.session_state.get("cropped_img_draw", None)
                        if cropped_img_draw:
                            cropped_array = np.array(cropped_img_draw)
                            if len(cropped_array.shape) == 3:
                                cropped_cv = cv2.cvtColor(cropped_array, cv2.COLOR_RGB2BGR)
                            else:
                                cropped_cv = cropped_array
                            
                            st.session_state.original_img = cropped_cv.copy()
                            st.session_state.working_img = cropped_cv.copy()
                            st.session_state.stage = 4
                            st.rerun()
                        else:
                            st.warning("Please draw a cropping area first.")
                    else:  # coordinates
                        # Crop using coordinates
                        cropped_array = np.array(preview_img)
                        if len(cropped_array.shape) == 3:
                            cropped_cv = cv2.cvtColor(cropped_array, cv2.COLOR_RGB2BGR)
                        else:
                            cropped_cv = cropped_array
                        
                        st.session_state.original_img = cropped_cv.copy()
                        st.session_state.working_img = cropped_cv.copy()
                        st.session_state.stage = 4
                        st.rerun()
        
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
    Two-column layout:
      - Left: image / hand selection area
      - Right: options (original, hand crop, coordinate crop) with previews and navigation.
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

    # Remember last selected mode
    default_mode = st.session_state.get("crop_mode", "Original image")

    # Prepare containers
    left_col, right_col = st.columns([3, 2])

    # -----------------------------
    # Right column: options & settings
    # -----------------------------
    with right_col:
        # Header line: adjust based on current mode from session state.
        header_mode = st.session_state.get("crop_mode", "Original image")
        if header_mode == "Original image":
            st.markdown("### Crop Options")
        else:
            # 24 non-breaking spaces between labels
            st.markdown(
                "### Crop Options&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Crop Preview",
                unsafe_allow_html=True,
            )

        # Top row: options on the left, preview on the right with a small horizontal gap.
        # The options column is slightly narrower so the preview label has room on one line.
        opt_col, preview_col = st.columns([1, 1.5])

        with opt_col:
            mode = st.radio(
                "Mode:",
                ["Original image", "Hand crop", "Coordinate crop"],
                key="crop_mode",
                index=["Original image", "Hand crop", "Coordinate crop"].index(default_mode)
                if default_mode in ["Original image", "Hand crop", "Coordinate crop"]
                else 0,
            )
            # Limit the info box width to the options column only
            st.info(f"**Original Dimensions:** {original_width} × {original_height} px")
            # Placeholder for crop dimensions - will be filled after images are computed
            crop_dims_placeholder = st.empty()

        # Container where the preview will appear (to the right of options),
        # wrapped in a small left margin to approximate a 5px gap.
        with preview_col:
            st.markdown(
                "<div style='margin-left:5px;'>",
                unsafe_allow_html=True,
            )
            preview_container = st.container()
            st.markdown("</div>", unsafe_allow_html=True)

        # Default values for hand crop configuration
        hand_realtime_update = True
        hand_box_color = "#FF0000"
        hand_aspect_ratio_value = None

        # Coordinate crop outputs
        coord_cropped_img = None
        coord_box = None

        # Settings dropdowns below the preview
        if mode == "Hand crop":
            with st.expander("Hand Crop Settings", expanded=True):
                hand_realtime_update = st.checkbox(
                    "Update in real time",
                    value=st.session_state.get("crop_realtime", True),
                    key="crop_realtime",
                )
                hand_box_color = st.color_picker(
                    "Box color",
                    value=st.session_state.get("crop_box_color", "#FF0000"),
                    key="crop_box_color",
                )
                aspect_ratio = st.selectbox(
                    "Aspect ratio",
                    ["Free", "1:1", "4:3", "16:9", "3:2"],
                    key="crop_aspect_ratio",
                )
                aspect_dict = {
                    "Free": None,
                    "1:1": (1, 1),
                    "4:3": (4, 3),
                    "16:9": (16, 9),
                    "3:2": (3, 2),
                }
                hand_aspect_ratio_value = aspect_dict[aspect_ratio]
        elif mode == "Coordinate crop":
            with st.expander("Coordinate Crop Settings", expanded=True):
                st.markdown(
                    "Define a bounding box in **pixels** using a bottom-left origin "
                    "(0, 0) at the bottom-left of the image."
                )

                # User-defined bounding box dimensions
                bbox_width = st.number_input(
                    "Box width (px):",
                    min_value=1,
                    max_value=original_width,
                    value=min(256, original_width),
                    step=1,
                    key="coord_bbox_width",
                )
                bbox_height = st.number_input(
                    "Box height (px):",
                    min_value=1,
                    max_value=original_height,
                    value=min(256, original_height),
                    step=1,
                    key="coord_bbox_height",
                )

                st.markdown("**Bottom-left corner coordinates (origin at image bottom-left)**")
                coord_x = st.number_input(
                    "X (px from left):",
                    min_value=0,
                    max_value=max(0, original_width - 1),
                    value=0,
                    step=1,
                    key="coord_x",
                )
                coord_y = st.number_input(
                    "Y (px from bottom):",
                    min_value=0,
                    max_value=max(0, original_height - 1),
                    value=0,
                    step=1,
                    key="coord_y",
                )

                # Compute the cropped area and store the box for visualization
                coord_cropped_img, coord_box = _crop_from_bottom_left(
                    pil_img,
                    original_width,
                    original_height,
                    bbox_width,
                    bbox_height,
                    coord_x,
                    coord_y,
                )

    # -----------------------------
    # Left column: image / cropper
    # -----------------------------
    hand_cropped_img = None
    with left_col:
        st.markdown("### Select Area / View Image")
        if mode == "Hand crop":
            hand_cropped_img = st_cropper(
                pil_img,
                realtime_update=hand_realtime_update,
                box_color=hand_box_color,
                aspect_ratio=hand_aspect_ratio_value,
                key="image_cropper",
            )
            if hand_cropped_img is not None:
                hw, hh = hand_cropped_img.size
                st.caption(f"Hand crop selection: {hw} × {hh} px")
        elif mode == "Coordinate crop":
            # Show the original image with a visual bounding box overlay
            overlay_img = img_rgb.copy()
            if coord_box is not None:
                left_box, top_box, right_box, bottom_box = coord_box
                # Draw a green rectangle to represent the bounding box
                cv2.rectangle(
                    overlay_img,
                    (int(left_box), int(top_box)),
                    (int(right_box), int(bottom_box)),
                    (0, 255, 0),
                    2,
                )
            st.image(
                overlay_img,
                caption=f"Coordinate box overlay ({original_width} × {original_height} px)",
                use_container_width=True,
            )
        else:
            # Display original image using the same column width as the hand crop view
            st.image(
                pil_img,
                caption=f"Original image ({original_width} × {original_height} px)",
                use_container_width=True,
            )

    # -----------------------------
    # Update crop dimensions placeholder (after images are computed)
    # -----------------------------
    # Update the crop dimensions box in opt_col position
    if mode == "Hand crop" and hand_cropped_img is not None:
        hw, hh = hand_cropped_img.size
        crop_dims_placeholder.info(f"**Crop Dimensions:** {hw} × {hh} px")
    elif mode == "Coordinate crop" and coord_cropped_img is not None:
        cw, ch = coord_cropped_img.size
        crop_dims_placeholder.info(f"**Crop Dimensions:** {cw} × {ch} px")
    else:
        # Clear the placeholder if no crop is selected
        crop_dims_placeholder.empty()

    # -----------------------------
    # Right column: hand/coordinate preview image (to the right of options)
    # -----------------------------
    # Previews are rendered in the preview_container.
    with preview_container:
        if mode == "Hand crop" and hand_cropped_img is not None:
            st.image(
                hand_cropped_img,
                caption="Hand crop preview",
                use_container_width=True,
            )
        elif mode == "Coordinate crop" and coord_cropped_img is not None:
            st.image(
                coord_cropped_img,
                caption="Coordinate crop preview",
                use_container_width=True,
            )

    # -----------------------------
    # Bottom navigation buttons
    # -----------------------------
    st.markdown("---")
    nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 1])

    with nav_col1:
        if st.button("← Back to Upload", use_container_width=True):
            st.session_state.stage = 2
            st.rerun()

    with nav_col3:
        if st.button("Proceed →", use_container_width=True, type="primary"):
            selected_img = None

            if mode == "Original image":
                selected_img = pil_img
            elif mode == "Hand crop":
                if hand_cropped_img is not None:
                    selected_img = hand_cropped_img
                else:
                    st.warning("Please make a hand crop selection before proceeding.")
            elif mode == "Coordinate crop":
                if coord_cropped_img is not None:
                    selected_img = coord_cropped_img
                else:
                    st.warning("Please provide valid coordinate crop settings before proceeding.")

            if selected_img is not None:
                # Convert PIL image back to OpenCV format
                selected_array = np.array(selected_img)
                if len(selected_array.shape) == 3:
                    selected_cv = cv2.cvtColor(selected_array, cv2.COLOR_RGB2BGR)
                else:
                    selected_cv = selected_array

                st.session_state.original_img = selected_cv.copy()
                st.session_state.working_img = selected_cv.copy()
                st.session_state.stage = 4
                st.rerun()


def _crop_from_bottom_left(
    pil_img,
    original_width,
    original_height,
    width,
    height,
    x_from_left,
    y_from_bottom,
):
    """
    Crop image using a bounding box defined from the bottom-left corner.

    - Origin (0, 0) is at the bottom-left of the image.
    - x_from_left: pixels from the left edge.
    - y_from_bottom: pixels from the bottom edge.
    - width, height: size of the box in pixels.
    """
    # Clamp width/height to image size
    width = max(1, min(width, original_width))
    height = max(1, min(height, original_height))

    # Clamp coordinates so box stays within image bounds in bottom-left space
    max_x = max(0, original_width - width)
    max_y = max(0, original_height - height)
    x_from_left = max(0, min(x_from_left, max_x))
    y_from_bottom = max(0, min(y_from_bottom, max_y))

    # Convert bottom-left coordinates to PIL/top-left coordinate system
    left = x_from_left
    right = left + width

    # In bottom-left coordinates, box vertical range is [y_from_bottom, y_from_bottom + height]
    # Top in PIL coordinates is image_height - (y_from_bottom + height)
    # Bottom in PIL coordinates is image_height - y_from_bottom
    top = original_height - (y_from_bottom + height)
    bottom = original_height - y_from_bottom

    # Final clamp in PIL space
    left = max(0, min(left, original_width))
    right = max(0, min(right, original_width))
    top = max(0, min(top, original_height))
    bottom = max(0, min(bottom, original_height))

    if right <= left or bottom <= top:
        st.warning("Coordinate crop is out of bounds. Please adjust the box or coordinates.")
        return None, None

    cropped_img = pil_img.crop((left, top, right, bottom))
    return cropped_img, (left, top, right, bottom)
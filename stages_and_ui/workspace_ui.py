import base64
import streamlit as st
import time
import numpy as np
import cv2
from app_context import create_button, basic_tool_component, display_tool_help, interpret_query
import tools.image_operations as imops
from PIL import Image
from streamlit_cropper import st_cropper
from app_context import get_segmentor
import stages_and_ui.line_profile_ui as line_profile_ui

def loading_animation():
    gif_url = (
        "https://github.com/jmgerac/EM-Caddie/blob/main/assets/"
        "video_0_to_4s_slowstart_cropped_transparent.gif?raw=true"
    )

    st.markdown(
        f"""
        <div style="display:flex; align-items:center; justify-content:center; gap:50px;">
            <h1 style="font-size:80px; margin:0;">EM Caddie</h1>
            <img src="{gif_url}" width="300" />
        </div>
        """,
        unsafe_allow_html=True,
    )

    time.sleep(0.7)
    st.session_state.stage = 2
    st.rerun()



def initial_upload():
    st.markdown("<h2 style='text-align:center;'>EM Caddie</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Upload an image to get started</p>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"], key="startup_upload",
                                     label_visibility="collapsed")

    if uploaded_file:
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        cv2_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.session_state.original_img = cv2_img
        st.session_state.working_img = cv2_img.copy()
        st.session_state.stage = 3  # Go to cropping stage
        st.rerun()

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
    
    # Get cropped dimensions from cropper and sync with inputs
    # Track the previous cropper dimensions to detect when cropper moves
    if "prev_cropper_width" not in st.session_state:
        st.session_state.prev_cropper_width = None
        st.session_state.prev_cropper_height = None
    
    # Check if cropper result changed (user moved the box)
    cropper_changed = False
    if cropped_img:
        cropped_width, cropped_height = cropped_img.size
        if (st.session_state.prev_cropper_width != cropped_width or 
            st.session_state.prev_cropper_height != cropped_height):
            cropper_changed = True
            st.session_state.prev_cropper_width = cropped_width
            st.session_state.prev_cropper_height = cropped_height
        
        # Only update session state if user hasn't manually set dimensions
        # This allows the cropper to update the inputs, but manual input takes precedence
        if "manual_crop_set" not in st.session_state or not st.session_state.manual_crop_set:
            st.session_state.crop_width = cropped_width
            st.session_state.crop_height = cropped_height
    
    # Display editable dimension inputs (clickable to input pixel sizes)
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
                st.session_state.prev_cropper_width = None
                st.session_state.prev_cropper_height = None
                st.rerun()
    
    # Check if dimensions were manually changed via number inputs
    # Only treat as manual change if user actually interacted with inputs AND cropper didn't just change
    dimensions_changed = (input_width != st.session_state.crop_width or 
                         input_height != st.session_state.crop_height)
    
    # Prioritize cropper result if cropper changed and manual mode is not set
    if cropper_changed and ("manual_crop_set" not in st.session_state or not st.session_state.manual_crop_set):
        # Use the cropper result - cropped_img is already set from st_cropper
        pass  # cropped_img is already the correct value from st_cropper
    elif dimensions_changed and not cropper_changed:
        # User manually changed dimensions via number inputs
        st.session_state.crop_width = input_width
        st.session_state.crop_height = input_height
        st.session_state.manual_crop_set = True
        
        # Crop from bottom-right corner
        left = max(0, original_width - input_width)
        top = max(0, original_height - input_height)
        right = min(left + input_width, original_width)
        bottom = min(top + input_height, original_height)
        
        # Adjust if we're at the edge
        if right - left < input_width:
            left = max(0, right - input_width)
        if bottom - top < input_height:
            top = max(0, bottom - input_height)
        
        # Create cropped image from specified dimensions
        cropped_img = pil_img.crop((left, top, right, bottom))
        st.info(f"✅ Cropped from bottom-right corner: ({left}, {top}) to ({right}, {bottom})")
    elif cropped_img and ("manual_crop_set" not in st.session_state or not st.session_state.manual_crop_set):
        # Use the cropper result if no manual input was set
        pass  # cropped_img is already the correct value from st_cropper
    elif st.session_state.get("manual_crop_set", False):
        # Use manual dimensions if they were previously set
        left = max(0, original_width - st.session_state.crop_width)
        top = max(0, original_height - st.session_state.crop_height)
        right = min(left + st.session_state.crop_width, original_width)
        bottom = min(top + st.session_state.crop_height, original_height)
        cropped_img = pil_img.crop((left, top, right, bottom))
    
    # Display the cropped image
    if cropped_img:
        # Get actual dimensions of cropped image
        final_width, final_height = cropped_img.size
        
        # Show preview of cropped image
        st.markdown("### Preview:")
        st.image(cropped_img, caption=f"Cropped Image ({final_width} × {final_height} pixels)", use_container_width=True)
        
        # Action buttons
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("← Back to Upload", use_container_width=True):
                st.session_state.stage = 2
                st.rerun()
        
        with col2:
            if st.button("Use Original Image", use_container_width=True):
                # Keep original image
                st.session_state.working_img = st.session_state.original_img.copy()
                st.session_state.stage = 4  # Go to workspace
                st.rerun()
        
        with col3:
            if st.button("Use Cropped Image →", use_container_width=True, type="primary"):
                # Convert PIL image back to OpenCV format
                cropped_array = np.array(cropped_img)
                if len(cropped_array.shape) == 3:
                    cropped_cv = cv2.cvtColor(cropped_array, cv2.COLOR_RGB2BGR)
                else:
                    cropped_cv = cropped_array
                
                # Update both original and working images to cropped version
                st.session_state.original_img = cropped_cv.copy()
                st.session_state.working_img = cropped_cv.copy()
                st.session_state.stage = 4  # Go to workspace
                st.rerun()
    else:
        st.warning("Please select a cropping area.")


def set_working_from_index():
    st.session_state.working_img = (
        st.session_state.timeline_images[st.session_state.timeline_index]
    )


def undo_cb():
    if st.session_state.timeline_index > 0:
        st.session_state.timeline_index -= 1
        set_working_from_index()


def redo_cb():
    if st.session_state.timeline_index < len(st.session_state.timeline):
        st.session_state.timeline_index += 1
        set_working_from_index()


def reset_cb():
    orig = st.session_state.original_img.copy()
    st.session_state.timeline = []
    st.session_state.timeline_index = 0
    st.session_state.timeline_images = [orig]
    st.session_state.working_img = orig


def apply_pipeline_cb(tools, scale_bar_params=None, segmentation_params=None):
    if not st.session_state.pipeline:
        return

    idx = st.session_state.timeline_index

    # Truncate undone future
    st.session_state.timeline = st.session_state.timeline[:idx]
    st.session_state.timeline_images = st.session_state.timeline_images[: idx + 1]

    img = st.session_state.timeline_images[-1]

    for tool_name in st.session_state.pipeline:
        # Apply tool
        if tool_name == "Add Scale Bar":
            if scale_bar_params and isinstance(scale_bar_params, dict):
                img = imops.add_scale_bar(
                    img.copy(),
                    scale_bar_params["scale_bar_length"],
                    units=scale_bar_params["units"],
                    input_mode=scale_bar_params["input_mode"],
                    pixel_to_scale=scale_bar_params.get("pixel_to_scale"),
                    total_image_length=scale_bar_params.get("total_image_length"),
                    position=scale_bar_params["position"],
                    x_offset=scale_bar_params.get("x_offset"),
                    y_offset=scale_bar_params.get("y_offset"),
                    font_scale_factor=scale_bar_params["font_scale_factor"],
                    show_background=scale_bar_params.get("show_background", True),
                    bar_color=scale_bar_params.get("bar_color", "#FFFFFF"),
                    font_color=scale_bar_params.get("font_color", "#FFFFFF"),
                )
            else:
                # Fallback defaults
                img = imops.add_scale_bar(
                    img.copy(),
                    10.0,
                    units="nm",
                    input_mode="total_length",
                    total_image_length=100.0,
                )
        else:
            if tool_name == "AtomAI Segmentation":
                segmentor = get_segmentor()
                prob_mask = imops.atomai_segment(img.copy(), segmentor)

                # Run model once
                prob_mask = imops.atomai_segment(img.copy(), segmentor)

                # Apply UI choice
                if segmentation_params:
                    if segmentation_params["mode"] == "Binary threshold":
                        _, img = cv2.threshold(
                            prob_mask,
                            segmentation_params["threshold"],
                            255,
                            cv2.THRESH_BINARY,
                        )
                    else:
                        img = prob_mask
                else:
                    img = prob_mask

            else:
                tool_fn = tools[tool_name][1]
                if tool_fn is not None:
                    img = tool_fn(img.copy())

        # Record timeline step
        st.session_state.timeline.append(tool_name)
        st.session_state.timeline_images.append(img)

    st.session_state.timeline_index = len(st.session_state.timeline)
    st.session_state.working_img = img



def workspace(tools, tool_names, tool_embs, encoder):
    """
    Layout-safe, rerun-stable Streamlit workspace.
    Assumes this function is ONLY called when stage == 4.
    Streamlit workspace with:
    - timeline-based undo/redo
    - cached images for fast undo/redo
    - pipelines of multiple tools
    - sidebar showing applied/undone operation names
    """

    # -----------------------------
    # Initialize timeline + cached images
    # -----------------------------
    if "timeline" not in st.session_state:
        st.session_state.timeline = []  # list of applied tool names
    if "timeline_index" not in st.session_state:
        st.session_state.timeline_index = 0  # how many ops currently applied
    if "timeline_images" not in st.session_state:
        st.session_state.timeline_images = [st.session_state.original_img.copy()]
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = []

    # -----------------------------
    # Sidebar: manual tool select + history
    # -----------------------------
    # Button to return to the cropping stage
    if st.sidebar.button("← Back to Cropping", use_container_width=True, type="secondary"):
        st.session_state.stage = 3
        st.rerun()
    
    # Add button to go to image analysis workshop
    if st.sidebar.button("📊 Open Image Analysis Workshop", use_container_width=True, type="secondary"):
        st.session_state.stage = 5
        st.rerun()
    
    selected_tool = st.sidebar.selectbox(
        "Pick a tool manually:",
        [""] + tool_names,
        key="manual_tool_select",
    )

    st.sidebar.markdown(f"### Timeline ( index = {st.session_state.timeline_index} )")

    # Display timeline with applied/undone status
    for i, (tool_name, img) in enumerate(
        zip(["Original Image"] + st.session_state.timeline, st.session_state.timeline_images)
    ):
        # small preview image
        _, buf = cv2.imencode(".png", img)
        preview_img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)

        # show grayscale/color thumbnail, fixed width
        st.sidebar.image(
            preview_img,
            width=60,
            caption=tool_name if i <= st.session_state.timeline_index else f"{tool_name} (undone)",
        )

    # -----------------------------
    # Main layout
    # -----------------------------
    col_main, col_right = st.columns([4, 2])

    # -----------------------------
    # Right column: controls
    # -----------------------------
    with col_right:
        query = st.text_area(
            "What would you like to do today?",
            "",
            height=200,
            key="tool_query_input",
        )

        col_undo, col_redo, _, col_reset = st.columns([1, 1, 0.5, 1])

        # Disable undo/redo if not available
        with col_undo:
            create_button(
                "↶",
                key="undo_btn",
                disabled=st.session_state.timeline_index == 0,
                on_click=undo_cb,
            )

        with col_redo:
            create_button(
                "↷",
                key="redo_btn",
                disabled=st.session_state.timeline_index == len(st.session_state.timeline),
                on_click=redo_cb,
            )

        with col_reset:
            create_button(
                "Clear",
                key="reset_btn",
                on_click=reset_cb,
            )


        # --- tool / pipeline suggestion ---
        pipeline = []
        if selected_tool:
            pipeline = [selected_tool]
        elif query:
            pipeline = [t for t in interpret_query(query) if t]

        st.session_state.pipeline = pipeline

        if pipeline:
            st.markdown("### Suggested pipeline")
            for i, tool in enumerate(pipeline, 1):
                st.markdown(f"{i}. **{tool}**")
        else:
            display_tool_help()

        # Check if scale bar is in pipeline and show button to open popover
        scale_bar_in_pipeline = "Add Scale Bar" in pipeline
        scale_bar_params = st.session_state.get("scale_bar_params", None)

        # Check if atomai in pipeline
        segmentation_in_pipeline = "AtomAI Segmentation" in pipeline
        segmentation_params = None
        
        if scale_bar_in_pipeline:
            # Button to open scale bar settings popover
            with st.popover("⚙️ Configure Scale Bar Settings", use_container_width=True):
                st.markdown("### Scale Bar Settings")
                
                # Create two columns: preview on left, settings on right
                preview_col, settings_col = st.columns([1.2, 1])
                
                with settings_col:
                    with st.container(border=True):
                        # Custom units input
                        units = st.text_input(
                            "Units:",
                            value=st.session_state.get("scale_bar_units", "nm"),
                            key="scale_bar_units",
                            help="Enter the unit for your scale bar (e.g., nm, μm, mm, px)"
                        )
                        
                        # Input mode selection
                        input_mode = st.radio(
                            "Input Mode:",
                            ["Total Image Length", "Pixel to Scale"],
                            index=0 if st.session_state.get("scale_bar_input_mode", "Total Image Length") == "Total Image Length" else 1,
                            key="scale_bar_input_mode",
                            help="Choose how to specify the scale"
                        )
                        
                        if input_mode == "Total Image Length":
                            total_image_length = st.number_input(
                                "Total image length:",
                                min_value=0.1,
                                value=st.session_state.get("scale_bar_total_length", 100.0),
                                step=1.0,
                                key="scale_bar_total_length",
                                help=f"The total length of the image in {units}"
                            )
                            pixel_to_scale = None
                        else:  # Pixel to Scale
                            pixel_to_scale = st.number_input(
                                "Length per pixel:",
                                min_value=0.0001,
                                value=st.session_state.get("scale_bar_pixel_to_scale", 0.1),
                                step=0.01,
                                format="%.4f",
                                key="scale_bar_pixel_to_scale",
                                help=f"Length per pixel in {units}/pixel"
                            )
                            total_image_length = None
                        
                        # Scale bar length
                        scale_length = st.number_input(
                            f"Scale bar length ({units}):",
                            min_value=0.1,
                            value=st.session_state.get("scale_bar_length", 10.0),
                            step=1.0,
                            key="scale_bar_length",
                            help=f"The desired length of the scale bar in {units}"
                        )
                        
                        # Position selection
                        position = st.selectbox(
                            "Position:",
                            ["Bottom Right", "Bottom Left", "Top Right", "Top Left", "Custom"],
                            index=["Bottom Right", "Bottom Left", "Top Right", "Top Left", "Custom"].index(
                                st.session_state.get("scale_bar_position", "Bottom Right")
                            ),
                            key="scale_bar_position",
                            help="Select the position of the scale bar"
                        )
                        
                        # Custom position controls
                        if position == "Custom":
                            st.markdown("**Position Controls:** Move the scale bar across the entire image")
                            col_x, col_y = st.columns(2)
                            with col_x:
                                x_offset = st.slider(
                                    "Horizontal Position:",
                                    min_value=0.0,
                                    max_value=1.0,
                                    value=st.session_state.get("scale_bar_x_offset", 0.9),
                                    step=0.01,
                                    key="scale_bar_x_offset",
                                    help="0.0 = left edge, 1.0 = right edge. The scale bar will move horizontally across the image."
                                )
                            with col_y:
                                y_offset = st.slider(
                                    "Vertical Position:",
                                    min_value=0.0,
                                    max_value=1.0,
                                    value=st.session_state.get("scale_bar_y_offset", 0.95),
                                    step=0.01,
                                    key="scale_bar_y_offset",
                                    help="0.0 = top edge, 1.0 = bottom edge. The scale bar will move vertically across the image."
                                )
                            # Set position to "custom" for the function call
                            position_param = "custom"
                        else:
                            x_offset = None
                            y_offset = None
                            # Convert position string to function parameter
                            position_map = {
                                "Bottom Right": "bottom_right",
                                "Bottom Left": "bottom_left",
                                "Top Right": "top_right",
                                "Top Left": "top_left"
                            }
                            position_param = position_map[position]
                        
                        # Font size adjustment
                        font_scale_factor = st.slider(
                            "Font Size:",
                            min_value=0.5,
                            max_value=3.0,
                            value=st.session_state.get("scale_bar_font_scale", 1.0),
                            step=0.1,
                            key="scale_bar_font_scale",
                            help="Adjust the font size of the scale bar label"
                        )
                        
                        # Color and appearance settings
                        st.markdown("---")
                        st.markdown("**Appearance Settings:**")
                        
                        # Background toggle
                        show_background = st.checkbox(
                            "Show solid black background",
                            value=st.session_state.get("scale_bar_show_background", True),
                            key="scale_bar_show_background",
                            help="Toggle the solid black background box behind the scale bar and text"
                        )
                        
                        # Color pickers
                        col_bar, col_font = st.columns(2)
                        with col_bar:
                            bar_color = st.color_picker(
                                "Scale Bar Color:",
                                value=st.session_state.get("scale_bar_color", "#FFFFFF"),
                                key="scale_bar_color",
                                help="Choose the color of the scale bar"
                            )
                        with col_font:
                            font_color = st.color_picker(
                                "Font Color:",
                                value=st.session_state.get("scale_bar_font_color", "#FFFFFF"),
                                key="scale_bar_font_color",
                                help="Choose the color of the scale bar text"
                            )
                        
                        # Store parameters in session state
                        st.session_state.scale_bar_params = {
                            "scale_bar_length": scale_length,
                            "units": units,
                            "input_mode": "total_length" if input_mode == "Total Image Length" else "pixel_to_scale",
                            "pixel_to_scale": pixel_to_scale,
                            "total_image_length": total_image_length,
                            "position": position_param,
                            "x_offset": x_offset,
                            "y_offset": y_offset,
                            "font_scale_factor": font_scale_factor,
                            "show_background": show_background,
                            "bar_color": bar_color,
                            "font_color": font_color
                        }
                        scale_bar_params = st.session_state.scale_bar_params
                
                with preview_col:
                    # Preview section
                    st.markdown("### Preview")
                    preview_img = st.session_state.get("working_img")
                    if preview_img is not None:
                        # Create preview with scale bar (updates in real-time as sliders change)
                        preview_with_scale = imops.add_scale_bar(
                            preview_img.copy(),
                            scale_length,
                            units=units,
                            input_mode=scale_bar_params["input_mode"],
                            pixel_to_scale=pixel_to_scale,
                            total_image_length=total_image_length,
                            position=position_param,
                            x_offset=x_offset,
                            y_offset=y_offset,
                            font_scale_factor=font_scale_factor,
                            show_background=show_background,
                            bar_color=bar_color,
                            font_color=font_color
                        )
                        
                        # Convert for display
                        if len(preview_with_scale.shape) == 3:
                            preview_rgb = cv2.cvtColor(preview_with_scale, cv2.COLOR_BGR2RGB)
                        else:
                            preview_rgb = preview_with_scale
                        
                        st.image(preview_rgb, caption="Preview with Scale Bar", use_container_width=True)

        if segmentation_in_pipeline:
            st.markdown("---")
            st.markdown("### Segmentation Output")

            with st.container(border=True):

                seg_mode = st.radio(
                    "Output type:",
                    ["Probability mask", "Binary threshold"],
                    index=0,
                    key="seg_output_mode",
                )

                threshold = None
                if seg_mode == "Binary threshold":
                    threshold = st.slider(
                        "Threshold",
                        min_value=0,
                        max_value=255,
                        value=128,
                        step=1,
                        key="seg_threshold",
                    )

                segmentation_params = {
                    "mode": seg_mode,
                    "threshold": threshold,
                }
        # Get scale bar params for apply function
        apply_scale_bar_params = st.session_state.get("scale_bar_params", None) if scale_bar_in_pipeline else None
        
        apply_clicked = create_button(
            "Apply pipeline",
            key="apply_pipeline",
            on_click=apply_pipeline_cb,
            args=(tools, scale_bar_params, segmentation_params),
            disabled=not st.session_state.pipeline,
        )

    # -----------------------------
    # Main column: image + actions
    # -----------------------------
    with col_main:
        img = st.session_state.get("working_img")

        if img is None:
            st.info("No image loaded.")
            return

        # --- display image ---
        display_img = st.session_state.working_img

        # ---- HARD SAFETY CONVERSION ----
        if display_img.dtype != np.uint8:
            if display_img.max() <= 1.0:
                # float image in [0,1]
                display_img = (display_img * 255).astype(np.uint8)
            else:
                # float image in [0,255] or worse
                display_img = np.clip(display_img, 0, 255).astype(np.uint8)
        # --------------------------------

        if display_img.ndim == 3:
            display_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        else:
            display_rgb = display_img

        st.image(display_rgb, caption="Current Image", width="stretch")

        # --- download ---
        out = (
            display_img
            if len(display_img.shape) == 3
            else cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
        )
        _, buf = cv2.imencode(".png", out)

        st.download_button(
            "Download Result",
            data=buf.tobytes(),
            file_name="processed.png",
            mime="image/png",
        )


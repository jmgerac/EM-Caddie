import base64
import csv
import io
import streamlit as st
import numpy as np
import cv2
import plotly.graph_objects as go
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from streamlit.elements import image as st_image
import tools.line_profile_tool as lpt


def _extract_points_from_canvas(json_data):
    """
    Extracts a list of (x, y) points from the last drawn object on the canvas.
    Supports line and free-draw (path) objects from streamlit-drawable-canvas.
    """
    if not json_data or "objects" not in json_data:
        return []

    objects = json_data.get("objects") or []
    if not objects:
        return []

    obj = objects[-1]  # use the most recent stroke
    obj_type = obj.get("type")

    # Direct line object
    if obj_type == "line":
        return [
            (obj.get("x1", 0), obj.get("y1", 0)),
            (obj.get("x2", 0), obj.get("y2", 0)),
        ]

    # Free draw path
    path = obj.get("path")
    if path:
        pts = []
        for cmd in path:
            if len(cmd) >= 3 and cmd[0] in ("M", "L"):
                pts.append((cmd[1], cmd[2]))
        return pts

    return []


def _build_profile_csv(points, distances, intensities, sample_coords):
    """Create a CSV (as bytes) containing vertices, sampled coordinates, and intensities."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)

    writer.writerow(["vertex_index", "vertex_x", "vertex_y"])
    for idx, (x, y) in enumerate(points):
        writer.writerow([idx, x, y])

    writer.writerow([])  # spacer
    writer.writerow(["sample_index", "distance_px", "intensity", "sample_x", "sample_y"])
    for idx, (dist, inten, coord) in enumerate(zip(distances, intensities, sample_coords)):
        writer.writerow([idx, f"{dist:.3f}", inten, coord[0], coord[1]])

    return buffer.getvalue().encode("utf-8")


# -------------------------------------------------------------------
# Compatibility: streamlit-drawable-canvas expects image_to_url API.
# Some Streamlit versions have it with incompatible signatures, so we provide a compatible version.
# This function accepts variable arguments to handle different library call patterns.
# -------------------------------------------------------------------
def _compatible_image_to_url(image, width=None, clamp=False, channels="RGB", output_format="PNG", *args, **kwargs):
    """
    Compatibility wrapper for image_to_url that accepts variable arguments.
    The library may call this with different numbers of arguments depending on version.
    """
    try:
        from io import BytesIO
        import numpy as np
        from PIL import Image as PILImage
    except Exception:
        return None

    pil_img = None
    if isinstance(image, PILImage.Image):
        pil_img = image
    elif isinstance(image, np.ndarray):
        arr = image
        if arr.dtype != np.uint8:
            # Normalize to 0-255
            arr = arr.astype(np.float32)
            arr = np.clip(arr, 0, 1)
            arr = (arr * 255).astype(np.uint8)
        mode = "RGB"
        if arr.ndim == 2:
            mode = "L"
        elif arr.ndim == 3 and arr.shape[2] == 4:
            mode = "RGBA"
        pil_img = PILImage.fromarray(arr, mode=mode)

    if pil_img is None:
        return None

    if width:
        w_percent = width / float(pil_img.size[0])
        h_size = int(float(pil_img.size[1]) * w_percent)
        pil_img = pil_img.resize((width, h_size))

    buffer = BytesIO()
    pil_img.save(buffer, format=output_format)
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/{output_format.lower()};base64,{b64}"

# Always override to ensure compatibility with streamlit-drawable-canvas
st_image.image_to_url = _compatible_image_to_url

def line_profile_stage():
    """
    Interactive line profile analysis stage.
    """
    st.markdown("<h2 style='text-align:center;'>Line Profile & Shape Analysis Tool</h2>", unsafe_allow_html=True)
    
    img = st.session_state.get("working_img")
    if img is None:
        st.error("No image loaded. Please go back and upload an image.")
        if st.button("Back to Workspace"):
            st.session_state.stage = 4
            st.rerun()
        return
    
    # Initialize session state for line profile
    if "line_profile_mode" not in st.session_state:
        st.session_state.line_profile_mode = "line"  # "line", "shape", or "canvas"
    
    if "line_p1" not in st.session_state:
        st.session_state.line_p1 = None
    if "line_p2" not in st.session_state:
        st.session_state.line_p2 = None
    
    if "shape_type" not in st.session_state:
        st.session_state.shape_type = "circle"
    if "shape_center" not in st.session_state:
        st.session_state.shape_center = None
    if "shape_size" not in st.session_state:
        st.session_state.shape_size = 50
    if "shape_angle" not in st.session_state:
        st.session_state.shape_angle = 0
    if "shape_start_fraction" not in st.session_state:
        st.session_state.shape_start_fraction = 0.0
    if "shape_end_fraction" not in st.session_state:
        st.session_state.shape_end_fraction = 1.0
    
    if "line_angle" not in st.session_state:
        st.session_state.line_angle = 0
    if "line_length" not in st.session_state:
        st.session_state.line_length = 100
    if "line_center" not in st.session_state:
        st.session_state.line_center = None
    if "use_full_image_line" not in st.session_state:
        st.session_state.use_full_image_line = False
    if "full_line_vertical_offset" not in st.session_state:
        st.session_state.full_line_vertical_offset = 0.5
    
    # Get image dimensions and prepare for display
    if len(img.shape) == 3:
        height, width = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        height, width = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # Canvas background for drawable canvas - ensure proper format
    # Convert to uint8 if needed and ensure RGB mode
    if img_rgb.dtype != np.uint8:
        # Normalize to 0-255 range
        img_rgb = img_rgb.astype(np.float32)
        if img_rgb.max() > 1.0:
            img_rgb = np.clip(img_rgb, 0, 255)
        else:
            img_rgb = (img_rgb * 255).astype(np.uint8)
        img_rgb = img_rgb.astype(np.uint8)
    
    # Ensure the array is contiguous and in the right format
    if not img_rgb.flags['C_CONTIGUOUS']:
        img_rgb = np.ascontiguousarray(img_rgb)
    
    # Convert to PIL Image and ensure RGB mode
    pil_bg = Image.fromarray(img_rgb, mode='RGB')
    
    # Limit canvas size if image is too large (canvas has performance issues with very large images)
    max_canvas_size = 2000
    if width > max_canvas_size or height > max_canvas_size:
        # Calculate scaling factor to fit within max size while maintaining aspect ratio
        scale = min(max_canvas_size / width, max_canvas_size / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        pil_bg = pil_bg.resize((new_width, new_height), Image.Resampling.LANCZOS)
        # Update dimensions for canvas
        canvas_width = new_width
        canvas_height = new_height
    else:
        canvas_width = width
        canvas_height = height
    
    # Sidebar controls
    with st.sidebar:
        if st.button("← Back to Workspace", use_container_width=True):
            st.session_state.stage = 4
            st.rerun()
        
        st.markdown("### Mode Selection")
        mode = st.radio(
            "Analysis Mode:",
            ["Line Profile", "Shape Profile", "Canvas Draw"],
            index={"line": 0, "shape": 1, "canvas": 2}.get(st.session_state.line_profile_mode, 0),
            key="profile_mode_radio"
        )
        st.session_state.line_profile_mode = (
            "line" if mode == "Line Profile" else "shape" if mode == "Shape Profile" else "canvas"
        )
        
        st.markdown("---")
        
        if st.session_state.line_profile_mode == "line":
            st.markdown("### Line Controls")
            
            use_full = st.checkbox(
                "Full Image Line",
                value=st.session_state.use_full_image_line,
                key="full_line_checkbox",
                help="Create a line that spans the entire image"
            )
            st.session_state.use_full_image_line = use_full
            
            if use_full:
                angle = st.slider(
                    "Angle (degrees from bottom):",
                    min_value=0.0,
                    max_value=180.0,
                    value=float(st.session_state.line_angle),
                    step=1.0,
                    key="full_line_angle",
                    help="Angle relative to bottom of image (0 = horizontal, 90 = vertical)"
                )
                st.session_state.line_angle = angle
                
                vertical_offset = st.slider(
                    "Vertical Position:",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(st.session_state.get("full_line_vertical_offset", 0.5)),
                    step=0.01,
                    key="full_line_vertical_offset",
                    help="Vertical position of the line (0.0 = top, 0.5 = center, 1.0 = bottom)"
                )
                
                # Calculate full image line
                p1, p2 = lpt.get_full_image_line(img, angle, vertical_offset)
                st.session_state.line_p1 = p1
                st.session_state.line_p2 = p2
                
                st.markdown("---")
                st.markdown("**Click Instructions:**")
                st.info("Adjust the angle and vertical position sliders to position the full-image line. The intensity profile updates automatically.")
            else:
                # Manual line controls
                st.markdown("**Line Position:**")
                if st.session_state.line_center is None:
                    st.session_state.line_center = (width // 2, height // 2)
                
                center_x = st.slider(
                    "Center X:",
                    min_value=0,
                    max_value=width,
                    value=int(st.session_state.line_center[0]),
                    step=1,
                    key="line_center_x"
                )
                center_y = st.slider(
                    "Center Y:",
                    min_value=0,
                    max_value=height,
                    value=int(st.session_state.line_center[1]),
                    step=1,
                    key="line_center_y"
                )
                st.session_state.line_center = (center_x, center_y)
                
                length = st.slider(
                    "Length (pixels):",
                    min_value=10,
                    max_value=min(width, height),
                    value=int(st.session_state.line_length),
                    step=1,
                    key="line_length_slider"
                )
                st.session_state.line_length = length
                
                angle = st.slider(
                    "Angle (degrees from bottom):",
                    min_value=0.0,
                    max_value=360.0,
                    value=float(st.session_state.line_angle),
                    step=1.0,
                    key="line_angle_slider"
                )
                st.session_state.line_angle = angle
                
                # Calculate line endpoints from sliders (will be overridden if manual coords are used)
                angle_rad = np.radians(angle)
                half_length = length / 2
                dx = half_length * np.cos(angle_rad)
                dy = -half_length * np.sin(angle_rad)  # Negative because y increases downward
                
                p1 = (center_x - dx, center_y - dy)
                p2 = (center_x + dx, center_y + dy)
                
                # Clamp to image bounds
                p1 = (max(0, min(width, p1[0])), max(0, min(height, p1[1])))
                p2 = (max(0, min(width, p2[0])), max(0, min(height, p2[1])))
                
                st.session_state.line_p1 = p1
                st.session_state.line_p2 = p2
                
                st.markdown("---")
                st.markdown("**Alternative: Manual Coordinates**")
                use_manual = st.checkbox("Use manual coordinate input", value=st.session_state.get("manual_coords_line", False), key="manual_coords_line")
                if use_manual:
                    col_p1, col_p2 = st.columns(2)
                    with col_p1:
                        st.markdown("**Point 1:**")
                        p1_x = st.number_input("X:", min_value=0, max_value=width, value=int(st.session_state.line_p1[0]) if st.session_state.line_p1 else width//4, key="p1_x")
                        p1_y = st.number_input("Y:", min_value=0, max_value=height, value=int(st.session_state.line_p1[1]) if st.session_state.line_p1 else height//2, key="p1_y")
                        st.session_state.line_p1 = (p1_x, p1_y)
                    with col_p2:
                        st.markdown("**Point 2:**")
                        p2_x = st.number_input("X:", min_value=0, max_value=width, value=int(st.session_state.line_p2[0]) if st.session_state.line_p2 else 3*width//4, key="p2_x")
                        p2_y = st.number_input("Y:", min_value=0, max_value=height, value=int(st.session_state.line_p2[1]) if st.session_state.line_p2 else height//2, key="p2_y")
                        st.session_state.line_p2 = (p2_x, p2_y)
                
                st.markdown("---")
                st.markdown("**Click Instructions:**")
                if st.session_state.get("manual_coords_line", False):
                    st.info("Manual coordinates are active. The intensity profile updates automatically as you change the coordinates.")
                else:
                    st.info("Use the sliders above or enable manual coordinates to define a line. The intensity profile updates automatically.")
        
        else:  # Shape mode
            st.markdown("### Shape Controls")
            
            shape_type = st.selectbox(
                "Shape Type:",
                ["circle", "triangle", "square", "pentagon", "hexagon", "heptagon", "octagon"],
                index=["circle", "triangle", "square", "pentagon", "hexagon", "heptagon", "octagon"].index(st.session_state.shape_type) if st.session_state.shape_type in ["circle", "triangle", "square", "pentagon", "hexagon", "heptagon", "octagon"] else 0,
                key="shape_type_select"
            )
            st.session_state.shape_type = shape_type
            
            if st.session_state.shape_center is None:
                st.session_state.shape_center = (width // 2, height // 2)
            
            st.markdown("**Position:**")
            center_x = st.slider(
                "Center X:",
                min_value=0,
                max_value=width,
                value=int(st.session_state.shape_center[0]),
                step=1,
                key="shape_center_x"
            )
            center_y = st.slider(
                "Center Y:",
                min_value=0,
                max_value=height,
                value=int(st.session_state.shape_center[1]),
                step=1,
                key="shape_center_y"
            )
            st.session_state.shape_center = (center_x, center_y)
            
            size_label = "Radius" if shape_type == "circle" else "Side Length"
            size = st.slider(
                f"{size_label} (pixels):",
                min_value=10,
                max_value=min(width, height) // 2,
                value=int(st.session_state.shape_size),
                step=1,
                key="shape_size_slider"
            )
            st.session_state.shape_size = size
            
            angle = st.slider(
                "Rotation Angle (degrees from bottom):",
                min_value=0.0,
                max_value=360.0,
                value=float(st.session_state.shape_angle),
                step=1.0,
                key="shape_angle_slider"
            )
            st.session_state.shape_angle = angle
            
            st.markdown("---")
            st.markdown("**Perimeter Range Selection:**")
            st.markdown("Select which portion of the shape perimeter to analyze (0.0 = start, 1.0 = full perimeter)")
            
            start_fraction = st.slider(
                "Start Position:",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.shape_start_fraction),
                step=0.01,
                key="shape_start_fraction",
                help="Starting position along the perimeter (0.0 = bottom point)"
            )
            
            end_fraction = st.slider(
                "End Position:",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.shape_end_fraction),
                step=0.01,
                key="shape_end_fraction",
                help="Ending position along the perimeter (1.0 = full perimeter)"
            )
            
            st.markdown("---")
            st.markdown("**Alternative: Manual Center Coordinates**")
            use_manual_shape = st.checkbox("Use manual coordinate input", value=st.session_state.get("manual_coords_shape", False), key="manual_coords_shape")
            if use_manual_shape:
                center_x_manual = st.number_input("Center X:", min_value=0, max_value=width, value=int(st.session_state.shape_center[0]), key="shape_center_x_manual")
                center_y_manual = st.number_input("Center Y:", min_value=0, max_value=height, value=int(st.session_state.shape_center[1]), key="shape_center_y_manual")
                st.session_state.shape_center = (center_x_manual, center_y_manual)
            else:
                # Update shape center from sliders if not using manual
                st.session_state.shape_center = (center_x, center_y)
            
            st.markdown("---")
            st.markdown("**Click Instructions:**")
            st.info("Use the sliders above or manual coordinates to position the shape. The intensity profile along the perimeter updates automatically.")
    
    # Main area: Image with interactive plotly (line/shape modes only)
    if st.session_state.line_profile_mode in ("line", "shape"):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Image with Overlay")
            
            # Create figure with image
            fig = go.Figure()
            
            # Add image
            fig.add_trace(go.Image(z=img_rgb))
            
            # Add line or shape overlay
            if st.session_state.line_profile_mode == "line" and st.session_state.line_p1 and st.session_state.line_p2:
                p1 = st.session_state.line_p1
                p2 = st.session_state.line_p2
                fig.add_trace(go.Scatter(
                    x=[p1[0], p2[0]],
                    y=[p1[1], p2[1]],
                    mode='lines+markers',
                    line=dict(color='lime', width=3),
                    marker=dict(size=10, color='lime'),
                    name='Line'
                ))
            elif st.session_state.line_profile_mode == "shape" and st.session_state.shape_center:
                # Draw shape outline
                center = st.session_state.shape_center
                size = st.session_state.shape_size
                angle = st.session_state.shape_angle
                shape_type = st.session_state.shape_type
                
                # Get perimeter points
                start_frac = st.session_state.shape_start_fraction
                end_frac = st.session_state.shape_end_fraction
                points = lpt.get_shape_perimeter_points(shape_type, center, size, angle, start_frac, end_frac)
                if len(points) > 0:
                    fig.add_trace(go.Scatter(
                        x=points[:, 0],
                        y=points[:, 1],
                        mode='lines',
                        line=dict(color='lime', width=2),
                        fill='none',
                        name='Shape'
                    ))
                    # Mark center
                    fig.add_trace(go.Scatter(
                        x=[center[0]],
                        y=[center[1]],
                        mode='markers',
                        marker=dict(size=10, color='red', symbol='x'),
                        name='Center'
                    ))
            
            # Update layout
            fig.update_layout(
                xaxis=dict(range=[0, width], scaleanchor="y", scaleratio=1),
                yaxis=dict(range=[height, 0]),  # Inverted y-axis for image coordinates
                width=800,
                height=int(800 * height / width),
                showlegend=False,
                margin=dict(l=0, r=0, t=0, b=0)
            )
            
            # Display plotly figure
            clicked_data = st.plotly_chart(fig, use_container_width=True, key="interactive_image")
            
            # Handle click events (if plotly returns click data)
            # Note: Streamlit's plotly_chart doesn't directly support click events in the same way
            # We'll use the sliders for now, but add a note about clicking
            
            # Alternative: Use streamlit-image-coordinates or create a custom component
            # For now, we'll rely on the sliders and provide instructions
        
        with col2:
            st.markdown("### Intensity Profile")
            
            # Calculate and display profile
            if st.session_state.line_profile_mode == "line" and st.session_state.line_p1 and st.session_state.line_p2:
                distances, intensities = lpt.get_line_profile(
                    img,
                    st.session_state.line_p1,
                    st.session_state.line_p2
                )
                
                # Create profile plot
                profile_fig = go.Figure()
                profile_fig.add_trace(go.Scatter(
                    x=distances,
                    y=intensities,
                    mode='lines',
                    line=dict(color='cyan', width=2),
                    name='Intensity'
                ))
                profile_fig.update_layout(
                    title="Line Profile",
                    xaxis_title="Distance (pixels)",
                    yaxis_title="Intensity",
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(profile_fig, use_container_width=True)
                
                # Statistics
                st.markdown("**Statistics:**")
                col_stat1, col_stat2 = st.columns(2)
                with col_stat1:
                    st.metric("Mean", f"{np.mean(intensities):.2f}")
                    st.metric("Max", f"{np.max(intensities):.2f}")
                with col_stat2:
                    st.metric("Min", f"{np.min(intensities):.2f}")
                    st.metric("Std Dev", f"{np.std(intensities):.2f}")
            
            elif st.session_state.line_profile_mode == "shape" and st.session_state.shape_center:
                distances, intensities = lpt.get_shape_perimeter_profile(
                    img,
                    st.session_state.shape_type,
                    st.session_state.shape_center,
                    st.session_state.shape_size,
                    st.session_state.shape_angle,
                    st.session_state.shape_start_fraction,
                    st.session_state.shape_end_fraction
                )
                
                # Create profile plot
                profile_fig = go.Figure()
                profile_fig.add_trace(go.Scatter(
                    x=distances,
                    y=intensities,
                    mode='lines',
                    line=dict(color='cyan', width=2),
                    name='Intensity'
                ))
                profile_fig.update_layout(
                    title=f"{st.session_state.shape_type.capitalize()} Perimeter Profile",
                    xaxis_title="Distance along perimeter (pixels)",
                    yaxis_title="Intensity",
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(profile_fig, use_container_width=True)
                
                # Statistics
                st.markdown("**Statistics:**")
                col_stat1, col_stat2 = st.columns(2)
                with col_stat1:
                    st.metric("Mean", f"{np.mean(intensities):.2f}")
                    st.metric("Max", f"{np.max(intensities):.2f}")
                with col_stat2:
                    st.metric("Min", f"{np.min(intensities):.2f}")
                    st.metric("Std Dev", f"{np.std(intensities):.2f}")
            else:
                st.info("Configure the line or shape above to see the intensity profile.")

    # -------------------------------------------------------
    # Canvas-based drawing (click 2 points or freehand draw)
    # -------------------------------------------------------
    if st.session_state.line_profile_mode == "canvas":
        st.markdown("---")
        st.subheader("Canvas Draw")
        st.caption("Draw directly on the image to measure an intensity profile without sliders.")
        canvas_col, canvas_plot_col = st.columns([2, 1])

        with canvas_col:
            draw_mode = st.radio(
                "Drawing mode",
                ["line", "freedraw"],
                index=0,
                key="canvas_draw_mode",
                horizontal=True,
                help="Use 'line' to click two points, or 'freedraw' to trace any path."
            )
            stroke_color = st.color_picker("Stroke color", value="#00ff00", key="canvas_stroke_color")
            stroke_width = st.slider("Stroke width", 1, 15, value=3, key="canvas_stroke_width")
            st.caption("Tip: The most recent stroke is used for the profile calculation.")

            # Display a test image to verify conversion worked
            with st.expander("🔍 Debug: Image Preview", expanded=False):
                st.image(pil_bg, caption=f"Background image ({pil_bg.size[0]}x{pil_bg.size[1]}, mode: {pil_bg.mode})", use_container_width=True)
                st.write(f"Image type: {type(pil_bg)}, Mode: {pil_bg.mode}, Size: {pil_bg.size}")
                # Also show as numpy array info
                img_array = np.array(pil_bg)
                st.write(f"Array shape: {img_array.shape}, dtype: {img_array.dtype}, min: {img_array.min()}, max: {img_array.max()}")
            
            # Try multiple approaches to ensure the image displays
            # First, ensure PIL image is in the correct format
            if pil_bg.mode != 'RGB':
                pil_bg = pil_bg.convert('RGB')
            
            # Try passing as numpy array if PIL doesn't work
            # Convert back to numpy for potential alternative approach
            img_for_canvas = np.array(pil_bg)
            
            canvas_result = st_canvas(
                fill_color="rgba(0, 0, 0, 0)",
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                background_image=pil_bg,  # Try PIL first
                update_streamlit=True,
                height=canvas_height,
                width=canvas_width,
                drawing_mode=draw_mode,
                key="line_profile_draw_canvas",
            )
            
            # If canvas_result is None or has no background, try numpy array
            if canvas_result is None or (hasattr(canvas_result, 'json_data') and canvas_result.json_data is None):
                st.warning("⚠️ Canvas may not have loaded the background image. Trying alternative format...")
                # Note: streamlit-drawable-canvas typically expects PIL Image, but we can try numpy
                # Actually, let's keep PIL but ensure it's properly formatted
                pass

            canvas_points = _extract_points_from_canvas(canvas_result.json_data if canvas_result else None)
            
            # Convert canvas coordinates back to original image coordinates if image was resized
            if canvas_points and (canvas_width != width or canvas_height != height):
                scale_x = width / canvas_width
                scale_y = height / canvas_height
                canvas_points = [(p[0] * scale_x, p[1] * scale_y) for p in canvas_points]

        with canvas_plot_col:
            st.markdown("### Drawn Intensity Profile")
            if canvas_points and len(canvas_points) >= 2:
                distances, intensities, sample_coords = lpt.get_polyline_profile(img, canvas_points)

                profile_fig = go.Figure()
                profile_fig.add_trace(go.Scatter(
                    x=distances,
                    y=intensities,
                    mode='lines',
                    line=dict(color='cyan', width=2),
                    name='Intensity'
                ))
                profile_fig.update_layout(
                    title="Profile from Canvas Path",
                    xaxis_title="Distance along path (pixels)",
                    yaxis_title="Intensity",
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(profile_fig, use_container_width=True)

                csv_bytes = _build_profile_csv(canvas_points, distances, intensities, sample_coords)
                st.download_button(
                    "Download CSV (points + intensity)",
                    data=csv_bytes,
                    file_name="drawn_intensity_profile.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            else:
                st.info("Draw a line or freehand stroke on the canvas to see its intensity profile here.")
    
    # Add instructions
    st.markdown("---")
    with st.expander("How to Use", expanded=False):
        st.markdown("""
        **Line Profile Mode:**
        1. Choose "Full Image Line" to create a line spanning the entire image, or uncheck it for a custom line
        2. Use the sliders to adjust the line position, length, and angle
        3. The intensity profile updates automatically as you adjust the controls
        4. For full image lines, only the angle needs to be set
        
        **Shape Profile Mode:**
        1. Select a shape type (circle, square, or triangle)
        2. Adjust the center position, size, and rotation angle using the sliders
        3. The intensity profile along the perimeter (starting from bottom, going right around) updates automatically
        
        **Note:** Click interactions on the plotly chart are limited in Streamlit. Use the sliders in the sidebar for precise control.
        """)


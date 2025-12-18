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


def _extract_points_from_object(obj):
    """
    Extracts points from a single canvas object.
    Supports line, free-draw (path), circle, and rectangle objects.
    """
    obj_type = obj.get("type")

    # Direct line object
    if obj_type == "line":
        return [
            (obj.get("x1", 0), obj.get("y1", 0)),
            (obj.get("x2", 0), obj.get("y2", 0)),
        ]

    # Circle object - extract perimeter points
    elif obj_type == "circle":
        # Canvas circle objects have left, top, radius, width, height
        # Center is at (left + width/2, top + height/2) or (left + radius, top + radius)
        left = obj.get("left", 0)
        top = obj.get("top", 0)
        radius = obj.get("radius", 0)
        # If radius not available, calculate from width/height
        if radius == 0:
            width = obj.get("width", 0)
            height = obj.get("height", 0)
            radius = min(width, height) / 2
            center_x = left + width / 2
            center_y = top + height / 2
        else:
            center_x = left + radius
            center_y = top + radius
        
        # Generate points around the circle perimeter
        num_points = max(32, int(2 * np.pi * radius))
        num_points = min(num_points, 360)
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        points = []
        for angle in angles:
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            points.append((x, y))
        return points

    # Rectangle object - extract perimeter points
    elif obj_type == "rect":
        left = obj.get("left", 0)
        top = obj.get("top", 0)
        width = obj.get("width", 0)
        height = obj.get("height", 0)
        
        # Generate points along the rectangle perimeter
        points = []
        # Top edge
        num_top = max(10, int(width))
        for x in np.linspace(left, left + width, num_top):
            points.append((x, top))
        # Right edge
        num_right = max(10, int(height))
        for y in np.linspace(top, top + height, num_right):
            points.append((left + width, y))
        # Bottom edge (reverse)
        num_bottom = max(10, int(width))
        for x in np.linspace(left + width, left, num_bottom):
            points.append((x, top + height))
        # Left edge (reverse)
        num_left = max(10, int(height))
        for y in np.linspace(top + height, top, num_left):
            points.append((left, y))
        return points

    # Free draw path
    elif obj_type == "path" or "path" in obj:
        path = obj.get("path")
        if path:
            pts = []
            for cmd in path:
                if len(cmd) >= 3 and cmd[0] in ("M", "L"):
                    pts.append((cmd[1], cmd[2]))
            return pts

    return []


def _extract_all_objects_from_canvas(json_data):
    """
    Extracts all objects from canvas with their colors and points.
    Returns a list of dicts with 'points', 'color', 'type', and 'index'.
    """
    if not json_data or "objects" not in json_data:
        return []

    objects = json_data.get("objects") or []
    if not objects:
        return []

    extracted = []
    colors = ["#00ff00", "#ff0000", "#0000ff", "#ffff00", "#ff00ff", "#00ffff", "#ff8800", "#8800ff"]
    
    for idx, obj in enumerate(objects):
        points = _extract_points_from_object(obj)
        if points:
            color = obj.get("stroke", colors[idx % len(colors)])
            extracted.append({
                'points': points,
                'color': color,
                'type': obj.get("type", "unknown"),
                'index': idx
            })
    
    return extracted


def _convert_to_units(distances_px, pixels_per_unit=1.0):
    """Convert pixel distances to units."""
    if pixels_per_unit == 1.0:
        return distances_px, "pixels"
    return distances_px / pixels_per_unit, st.session_state.get("scale_units", "units")


def _build_profile_csv(points, distances, intensities, sample_coords, pixels_per_unit=1.0, units="pixels"):
    """Create a CSV (as bytes) containing vertices, sampled coordinates, and intensities."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)

    writer.writerow(["vertex_index", "vertex_x", "vertex_y"])
    for idx, (x, y) in enumerate(points):
        writer.writerow([idx, x, y])

    writer.writerow([])  # spacer
    distances_units, units_label = _convert_to_units(distances, pixels_per_unit)
    writer.writerow(["sample_index", f"distance_{units_label}", "distance_px", "intensity", "sample_x", "sample_y"])
    for idx, (dist_px, inten, coord) in enumerate(zip(distances, intensities, sample_coords)):
        dist_units = distances_units[idx] if idx < len(distances_units) else dist_px / pixels_per_unit
        writer.writerow([idx, f"{dist_units:.3f}", f"{dist_px:.3f}", inten, coord[0], coord[1]])

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
    st.markdown("<h2 style='text-align:center;'>Image Analysis Workshop</h2>", unsafe_allow_html=True)
    
    img = st.session_state.get("working_img")
    if img is None:
        st.error("No image loaded. Please go back and upload an image.")
        if st.button("Return to Image Processing"):
            st.session_state.stage = 4
            st.rerun()
        return
    
    # Initialize session state for line profile - default to "freedraw" (Free Draw section)
    if "line_profile_mode" not in st.session_state:
        st.session_state.line_profile_mode = "freedraw"  # "freedraw", "line", or "shape"
    
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
    # Also scale canvas to fit nicely in the 2:1 column layout
    max_canvas_size = 2000
    
    # Calculate target height to match intensity profile column total height
    # Right column contains:
    # - Heading: ~30px
    # - Combined plot: 400px (from plotly fig height)
    # - Separators: ~20px
    # - Profile name inputs: ~40px per profile (estimate ~120px for typical use)
    # - Download button section: ~80px
    # Total estimated: ~650px
    plot_height = 400  # Height of the combined intensity profile plot
    overhead_height = 250  # Estimate for heading, inputs, buttons, spacing
    target_canvas_height = plot_height + overhead_height  # ~650px total
    
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
    
    # Scale canvas to target height while maintaining aspect ratio (for better fit with right column)
    # Scale both up and down to match the target height
    if canvas_height != target_canvas_height:
        scale_factor = target_canvas_height / canvas_height
        canvas_width = int(canvas_width * scale_factor)
        canvas_height = int(canvas_height * scale_factor)
        # Resize the PIL image to match
        pil_bg = pil_bg.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)
    
    # Sidebar controls
    with st.sidebar:
        if st.button("← Return to Image Processing", use_container_width=True):
            st.session_state.stage = 4
            st.rerun()
        
        st.markdown("### Mode Selection")
        mode = st.radio(
            "Analysis Mode:",
            ["Draw", "Line Profile", "Shape Profile"],
            index={"freedraw": 0, "line": 1, "shape": 2}.get(st.session_state.line_profile_mode, 0),
            key="profile_mode_radio"
        )
        st.session_state.line_profile_mode = (
            "freedraw" if mode == "Draw" else "line" if mode == "Line Profile" else "shape"
        )
        
        st.markdown("---")
        
        # Scale/Units input section
        st.markdown("### Scale & Units")
        use_scale = st.checkbox("Use scale conversion", value=st.session_state.get("use_scale", False), key="use_scale_checkbox")
        
        if use_scale:
            scale_length = st.number_input(
                "Image Length:",
                min_value=0.0,
                value=float(st.session_state.get("scale_length", 1.0)),
                step=0.1,
                key="scale_length_input",
                help="Total length of the image in the specified units"
            )
            st.session_state.scale_length = scale_length
            
            scale_units = st.text_input(
                "Units:",
                value=st.session_state.get("scale_units", "nm"),
                key="scale_units_input",
                help="Unit of measurement (e.g., nm, μm, mm)"
            )
            st.session_state.scale_units = scale_units
            
            # Calculate pixels per unit
            if scale_length > 0:
                pixels_per_unit = max(width, height) / scale_length
                st.session_state.pixels_per_unit = pixels_per_unit
                st.caption(f"Scale: {pixels_per_unit:.2f} pixels per {scale_units}")
            else:
                st.session_state.pixels_per_unit = 1.0
        else:
            st.session_state.pixels_per_unit = 1.0
            st.session_state.scale_units = "pixels"
        
        st.markdown("---")
        
        # Canvas controls for Draw mode
        if st.session_state.line_profile_mode == "freedraw":
            st.markdown("### Canvas Controls")
            draw_type = st.selectbox(
                "Draw Type:",
                ["line", "freedraw", "circle", "rectangle"],
                index=0,
                key="canvas_draw_type",
                help="Select what to draw: line (2 points), freehand, or shapes (click-drag to resize)"
            )
            
            # Map draw type to canvas drawing mode
            # The selectbox already stores the value in st.session_state.canvas_draw_type
            if draw_type in ["line", "freedraw"]:
                canvas_draw_mode = draw_type
            elif draw_type == "circle":
                canvas_draw_mode = "circle"
            elif draw_type == "rectangle":
                canvas_draw_mode = "rect"
            else:
                canvas_draw_mode = "line"
            
            # Store the canvas drawing mode in session state
            st.session_state.canvas_draw_mode = canvas_draw_mode
            
            st.color_picker("Stroke color", value="#00ff00", key="canvas_stroke_color")
            st.slider("Stroke width", 1, 15, value=1, key="canvas_stroke_width")
            st.caption("💡 Tip: The most recent stroke is used for the profile calculation.")
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
        
        elif st.session_state.line_profile_mode == "shape":  # Only show shape controls in shape mode
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
    
    # ============================================================
    # DRAW SECTION (Default - shown first)
    # ============================================================
    if st.session_state.line_profile_mode == "freedraw":
        st.markdown("## Draw")
        st.caption("Draw multiple lines, shapes, or freehand strokes on the image. Each drawing gets its own color and separate intensity profile. Use the dropdown to select draw type, then click-drag to create shapes.")
        
        # Get canvas control values from session state (set by sidebar widgets)
        draw_type = st.session_state.get("canvas_draw_type", "line")
        draw_mode = st.session_state.get("canvas_draw_mode", "line")
        stroke_color = st.session_state.get("canvas_stroke_color", "#00ff00")
        stroke_width = st.session_state.get("canvas_stroke_width", 3)
        
        # Main canvas area
        col_canvas, col_plot = st.columns([2, 1])
        
        with col_canvas:
            st.markdown("### Drawing Canvas")
            
            # Ensure PIL image is in the correct format
            if pil_bg.mode != 'RGB':
                pil_bg = pil_bg.convert('RGB')
            
            # Map draw type to canvas drawing mode
            # Note: streamlit-drawable-canvas supports: "freedraw", "line", "rect", "circle", "transform"
            # For triangle, we'll use transform mode or handle it as a custom shape
            if draw_type in ["line", "freedraw"]:
                canvas_drawing_mode = draw_type
            elif draw_type == "circle":
                canvas_drawing_mode = "circle"
            elif draw_type == "rectangle":
                canvas_drawing_mode = "rect"
            else:
                canvas_drawing_mode = "line"
            
            canvas_result = st_canvas(
                fill_color="rgba(0, 0, 0, 0)",
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                background_image=pil_bg,
                update_streamlit=True,
                height=canvas_height,
                width=canvas_width,
                drawing_mode=canvas_drawing_mode,
                key="line_profile_draw_canvas",
            )
            
            # Extract all objects from canvas
            all_objects = _extract_all_objects_from_canvas(canvas_result.json_data if canvas_result else None)
            
            # Convert canvas coordinates back to original image coordinates if image was resized
            scale_x = width / canvas_width if canvas_width != width else 1.0
            scale_y = height / canvas_height if canvas_height != height else 1.0
            
            # Process each object
            processed_objects = []
            pixels_per_unit = st.session_state.get("pixels_per_unit", 1.0)
            
            for obj_data in all_objects:
                points = obj_data['points']
                if len(points) >= 2:
                    # Scale coordinates if needed
                    if scale_x != 1.0 or scale_y != 1.0:
                        points = [(p[0] * scale_x, p[1] * scale_y) for p in points]
                    
                    distances, intensities, sample_coords = lpt.get_polyline_profile(img, points)
                    if len(distances) > 0:
                        # Initialize name if not exists
                        profile_name_key = f"profile_name_{obj_data['index']}"
                        default_name = f"Profile {obj_data['index'] + 1}"
                        if profile_name_key not in st.session_state:
                            st.session_state[profile_name_key] = default_name
                        
                        processed_objects.append({
                            'points': points,
                            'distances': distances,
                            'intensities': intensities,
                            'sample_coords': sample_coords,
                            'color': obj_data['color'],
                            'index': obj_data['index'],
                            'type': obj_data['type'],
                            'name': st.session_state[profile_name_key]
                        })
        
        with col_plot:
            st.markdown("### Intensity Profiles")
            pixels_per_unit = st.session_state.get("pixels_per_unit", 1.0)
            units_label = st.session_state.get("scale_units", "pixels")
            
            if processed_objects:
                # Combined plot with all profiles (moved before name entry boxes)
                combined_fig = go.Figure()
                
                for obj in processed_objects:
                    distances_units, _ = _convert_to_units(obj['distances'], pixels_per_unit)
                    combined_fig.add_trace(go.Scatter(
                        x=distances_units,
                        y=obj['intensities'],
                        mode='lines',
                        line=dict(color=obj['color'], width=2),
                        name=obj['name'],
                        legendgroup=f"group_{obj['index']}"
                    ))
                
                xaxis_title = f"Distance along path ({units_label})" if pixels_per_unit != 1.0 else "Distance along path (pixels)"
                combined_fig.update_layout(
                    title="All Intensity Profiles",
                    xaxis_title=xaxis_title,
                    yaxis_title="Intensity",
                    height=400,
                    showlegend=True
                )
                st.plotly_chart(combined_fig, use_container_width=True)
                
                st.markdown("---")
                
                # Name entry boxes for each profile (moved after combined plot)
                st.markdown("**Profile Names:**")
                for obj in processed_objects:
                    profile_name_key = f"profile_name_{obj['index']}"
                    default_name = f"Profile {obj['index'] + 1}"
                    # Initialize name in session state if not exists
                    if profile_name_key not in st.session_state:
                        st.session_state[profile_name_key] = default_name
                    
                    current_name = st.session_state[profile_name_key]
                    new_name = st.text_input(
                        f"Name for {obj['type']} {obj['index'] + 1}:",
                        value=current_name,
                        key=f"name_input_{obj['index']}",
                        label_visibility="visible"
                    )
                    st.session_state[profile_name_key] = new_name
                    # Update the name in processed_objects
                    obj['name'] = new_name
                
                # Combined CSV Download
                st.markdown("---")
                st.markdown("**Download Combined Data:**")
                # Create combined CSV with all profiles
                combined_csv_buffer = io.StringIO()
                combined_csv_writer = csv.writer(combined_csv_buffer)
                
                combined_csv_writer.writerow(["Profile Name", "Profile Index", "Profile Type"])
                for obj in processed_objects:
                    combined_csv_writer.writerow([obj['name'], obj['index'], obj['type']])
                
                combined_csv_writer.writerow([])  # spacer
                
                # Write all profile data
                for obj in processed_objects:
                    combined_csv_writer.writerow([f"=== {obj['name']} ==="])
                    combined_csv_writer.writerow(["vertex_index", "vertex_x", "vertex_y"])
                    for idx, (x, y) in enumerate(obj['points']):
                        combined_csv_writer.writerow([idx, x, y])
                    
                    combined_csv_writer.writerow([])
                    distances_units, _ = _convert_to_units(obj['distances'], pixels_per_unit)
                    combined_csv_writer.writerow(["sample_index", f"distance_{units_label}", "distance_px", "intensity", "sample_x", "sample_y"])
                    for idx, (dist_px, inten, coord) in enumerate(zip(obj['distances'], obj['intensities'], obj['sample_coords'])):
                        dist_units = distances_units[idx] if idx < len(distances_units) else dist_px / pixels_per_unit
                        combined_csv_writer.writerow([idx, f"{dist_units:.3f}", f"{dist_px:.3f}", inten, coord[0], coord[1]])
                    combined_csv_writer.writerow([])  # spacer between profiles
                
                combined_csv_bytes = combined_csv_buffer.getvalue().encode("utf-8")
                st.download_button(
                    "📥 Download Combined CSV (All Profiles)",
                    data=combined_csv_bytes,
                    file_name="all_intensity_profiles.csv",
                    mime="text/csv",
                    use_container_width=True,
                    help="Download all profiles in a single CSV file"
                )
                
                # Individual profiles with downloads (in expander, shown in sidebar area)
                # Initialize show_individual variable
                show_individual = st.session_state.get("show_individual_plots", False)
                if show_individual:
                    st.markdown("---")
                    st.markdown("### Individual Profile Details")
                    for obj in processed_objects:
                        with st.expander(f"{obj['name']} ({obj['type']})", expanded=False):
                            pixels_per_unit = st.session_state.get("pixels_per_unit", 1.0)
                            units_label = st.session_state.get("scale_units", "pixels")
                            
                            # Check if this is a circle
                            is_circle = obj['type'] == 'circle'
                            
                            if is_circle:
                                # For circles, convert to π units
                                # Calculate total perimeter length
                                total_perimeter = obj['distances'][-1] if len(obj['distances']) > 0 else 0
                                if total_perimeter > 0:
                                    # Convert distances to π units (0 to 2π)
                                    distances_pi = (obj['distances'] / total_perimeter) * 2 * np.pi
                                    
                                    profile_fig = go.Figure()
                                    profile_fig.add_trace(go.Scatter(
                                        x=distances_pi,
                                        y=obj['intensities'],
                                        mode='lines',
                                        line=dict(color=obj['color'], width=2),
                                        name='Intensity'
                                    ))
                                    
                                    # Create secondary x-axis with pixel/unit conversion
                                    distances_units, _ = _convert_to_units(obj['distances'], pixels_per_unit)
                                    
                                    profile_fig.update_layout(
                                        title=obj['name'],
                                        xaxis_title="Angle (π radians)",
                                        yaxis_title="Intensity",
                                        height=300,
                                        showlegend=False,
                                        xaxis=dict(
                                            tickmode='linear',
                                            tick0=0,
                                            dtick=np.pi/2,
                                            tickformat='.2f',
                                            ticktext=['0', 'π/2', 'π', '3π/2', '2π'],
                                            tickvals=[0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
                                        )
                                    )
                                    
                                    # Add secondary axis annotation showing pixel/unit values
                                    st.plotly_chart(profile_fig, use_container_width=True)
                                    
                                    # Show conversion table (using details instead of expander to avoid nesting)
                                    st.markdown("**Angle to Distance Conversion:**")
                                    st.markdown(f"- **Total Perimeter:** {total_perimeter:.2f} pixels")
                                    if pixels_per_unit != 1.0:
                                        perimeter_units = total_perimeter / pixels_per_unit
                                        st.markdown(f"- **Total Perimeter:** {perimeter_units:.2f} {units_label}")
                                    st.markdown("**Key Angles:**")
                                    key_angles = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
                                    for angle in key_angles:
                                        fraction = angle / (2 * np.pi)
                                        dist_px = total_perimeter * fraction
                                        dist_units, _ = _convert_to_units(np.array([dist_px]), pixels_per_unit)
                                        angle_label = ['0', 'π/2', 'π', '3π/2', '2π'][int(angle / (np.pi/2))]
                                        st.markdown(f"  - {angle_label}: {dist_px:.2f} px" + 
                                                   (f" ({dist_units[0]:.2f} {units_label})" if pixels_per_unit != 1.0 else ""))
                                else:
                                    # Fallback for zero-length circle
                                    distances_units, _ = _convert_to_units(obj['distances'], pixels_per_unit)
                                    profile_fig = go.Figure()
                                    profile_fig.add_trace(go.Scatter(
                                        x=distances_units,
                                        y=obj['intensities'],
                                        mode='lines',
                                        line=dict(color=obj['color'], width=2),
                                        name='Intensity'
                                    ))
                                    xaxis_title = f"Distance ({units_label})" if pixels_per_unit != 1.0 else "Distance (pixels)"
                                    profile_fig.update_layout(
                                        title=obj['name'],
                                        xaxis_title=xaxis_title,
                                        yaxis_title="Intensity",
                                        height=300,
                                        showlegend=False
                                    )
                                    st.plotly_chart(profile_fig, use_container_width=True)
                            else:
                                # For non-circles, use standard distance units
                                distances_units, _ = _convert_to_units(obj['distances'], pixels_per_unit)
                                
                                profile_fig = go.Figure()
                                profile_fig.add_trace(go.Scatter(
                                    x=distances_units,
                                    y=obj['intensities'],
                                    mode='lines',
                                    line=dict(color=obj['color'], width=2),
                                    name='Intensity'
                                ))
                                xaxis_title = f"Distance ({units_label})" if pixels_per_unit != 1.0 else "Distance (pixels)"
                                profile_fig.update_layout(
                                    title=obj['name'],
                                    xaxis_title=xaxis_title,
                                    yaxis_title="Intensity",
                                    height=300,
                                    showlegend=False
                                )
                                st.plotly_chart(profile_fig, use_container_width=True)
                        
                        # Statistics
                        col_stat1, col_stat2 = st.columns(2)
                        with col_stat1:
                            st.metric("Mean", f"{np.mean(obj['intensities']):.2f}")
                            st.metric("Max", f"{np.max(obj['intensities']):.2f}")
                        with col_stat2:
                            st.metric("Min", f"{np.min(obj['intensities']):.2f}")
                            st.metric("Std Dev", f"{np.std(obj['intensities']):.2f}")
                        
                        # CSV Download
                        csv_bytes = _build_profile_csv(
                            obj['points'], obj['distances'], obj['intensities'], 
                            obj['sample_coords'], pixels_per_unit, units_label
                        )
                        st.download_button(
                            f"📥 Download CSV - {obj['name']}",
                            data=csv_bytes,
                            file_name=f"{obj['name'].replace(' ', '_')}.csv",
                            mime="text/csv",
                            use_container_width=True,
                            key=f"download_csv_{obj['index']}"
                        )
            else:
                st.info("Draw lines, shapes, or freehand strokes on the canvas to see intensity profiles here.")
        
        # Show combined intensity plot below canvas in full width
        if processed_objects:
            st.markdown("---")
            
            # Checkbox to show individual plots (moved above full-width plot)
            show_individual = st.checkbox(
                "📊 Show Individual Plots",
                value=st.session_state.get("show_individual_plots", False),
                key="show_individual_plots_checkbox"
            )
            st.session_state.show_individual_plots = show_individual
            
            st.markdown("### Combined Intensity Profile (Full Width)")
            pixels_per_unit = st.session_state.get("pixels_per_unit", 1.0)
            units_label = st.session_state.get("scale_units", "pixels")
            
            full_profile_fig = go.Figure()
            has_circles = any(obj['type'] == 'circle' for obj in processed_objects)
            all_circles = all(obj['type'] == 'circle' for obj in processed_objects)
            
            for obj in processed_objects:
                is_circle = obj['type'] == 'circle'
                
                if is_circle and all_circles:
                    # For circles-only plots, use π units
                    total_perimeter = obj['distances'][-1] if len(obj['distances']) > 0 else 0
                    if total_perimeter > 0:
                        distances_pi = (obj['distances'] / total_perimeter) * 2 * np.pi
                        x_data = distances_pi
                        xaxis_title = "Angle (π radians)"
                    else:
                        distances_units, _ = _convert_to_units(obj['distances'], pixels_per_unit)
                        x_data = distances_units
                        xaxis_title = f"Distance ({units_label})" if pixels_per_unit != 1.0 else "Distance (pixels)"
                else:
                    # For non-circles or mixed plots, use distance units
                    distances_units, _ = _convert_to_units(obj['distances'], pixels_per_unit)
                    x_data = distances_units
                    xaxis_title = f"Distance along path ({units_label})" if pixels_per_unit != 1.0 else "Distance along path (pixels)"
                
                # Convert hex color to rgba for fill
                hex_color = obj['color'].lstrip('#')
                rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                full_profile_fig.add_trace(go.Scatter(
                    x=x_data,
                    y=obj['intensities'],
                    mode='lines',
                    line=dict(color=obj['color'], width=2),
                    name=obj['name'],
                    fill='tozeroy' if obj['index'] == 0 else None,
                    fillcolor=f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.1)" if obj['index'] == 0 else None
                ))
            
            # Set x-axis formatting for circles-only plots
            if all_circles:
                full_profile_fig.update_layout(
                    title="All Intensity Profiles",
                    xaxis_title="Angle (π radians)",
                    yaxis_title="Intensity",
                    height=500,
                    showlegend=True,
                    hovermode='x unified',
                    xaxis=dict(
                        tickmode='linear',
                        tick0=0,
                        dtick=np.pi/2,
                        tickformat='.2f',
                        ticktext=['0', 'π/2', 'π', '3π/2', '2π'],
                        tickvals=[0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
                    )
                )
            else:
                full_profile_fig.update_layout(
                    title="All Intensity Profiles",
                    xaxis_title=xaxis_title,
                    yaxis_title="Intensity",
                    height=500,
                    showlegend=True,
                    hovermode='x unified'
                )
            st.plotly_chart(full_profile_fig, use_container_width=True)
            
            # Show individual plots below if checkbox is checked
            if show_individual:
                st.markdown("---")
                st.markdown("### Individual Profile Plots")
                for obj in processed_objects:
                    st.markdown(f"#### {obj['name']}")
                    pixels_per_unit = st.session_state.get("pixels_per_unit", 1.0)
                    units_label = st.session_state.get("scale_units", "pixels")
                    
                    # Check if this is a circle
                    is_circle = obj['type'] == 'circle'
                    
                    if is_circle:
                        # For circles, convert to π units
                        total_perimeter = obj['distances'][-1] if len(obj['distances']) > 0 else 0
                        if total_perimeter > 0:
                            distances_pi = (obj['distances'] / total_perimeter) * 2 * np.pi
                            
                            profile_fig = go.Figure()
                            profile_fig.add_trace(go.Scatter(
                                x=distances_pi,
                                y=obj['intensities'],
                                mode='lines',
                                line=dict(color=obj['color'], width=2),
                                name='Intensity'
                            ))
                            
                            profile_fig.update_layout(
                                title=obj['name'],
                                xaxis_title="Angle (π radians)",
                                yaxis_title="Intensity",
                                height=400,
                                showlegend=False,
                                xaxis=dict(
                                    tickmode='linear',
                                    tick0=0,
                                    dtick=np.pi/2,
                                    tickformat='.2f',
                                    ticktext=['0', 'π/2', 'π', '3π/2', '2π'],
                                    tickvals=[0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
                                )
                            )
                            st.plotly_chart(profile_fig, use_container_width=True)
                        else:
                            distances_units, _ = _convert_to_units(obj['distances'], pixels_per_unit)
                            profile_fig = go.Figure()
                            profile_fig.add_trace(go.Scatter(
                                x=distances_units,
                                y=obj['intensities'],
                                mode='lines',
                                line=dict(color=obj['color'], width=2),
                                name='Intensity'
                            ))
                            xaxis_title = f"Distance ({units_label})" if pixels_per_unit != 1.0 else "Distance (pixels)"
                            profile_fig.update_layout(
                                title=obj['name'],
                                xaxis_title=xaxis_title,
                                yaxis_title="Intensity",
                                height=400,
                                showlegend=False
                            )
                            st.plotly_chart(profile_fig, use_container_width=True)
                    else:
                        # For non-circles, use standard distance units
                        distances_units, _ = _convert_to_units(obj['distances'], pixels_per_unit)
                        
                        profile_fig = go.Figure()
                        profile_fig.add_trace(go.Scatter(
                            x=distances_units,
                            y=obj['intensities'],
                            mode='lines',
                            line=dict(color=obj['color'], width=2),
                            name='Intensity'
                        ))
                        xaxis_title = f"Distance ({units_label})" if pixels_per_unit != 1.0 else "Distance (pixels)"
                        profile_fig.update_layout(
                            title=obj['name'],
                            xaxis_title=xaxis_title,
                            yaxis_title="Intensity",
                            height=400,
                            showlegend=False
                        )
                        st.plotly_chart(profile_fig, use_container_width=True)
                    
                    st.markdown("---")
    
    # ============================================================
    # LINE PROFILE SECTION
    # ============================================================
    elif st.session_state.line_profile_mode == "line":
        st.markdown("## Line Profile")
        st.caption("Use sliders or manual coordinates to define a line and analyze its intensity profile.")
    
    # ============================================================
    # SHAPE PROFILE SECTION
    # ============================================================
    elif st.session_state.line_profile_mode == "shape":
        st.markdown("## Shape Profile")
        st.caption("Analyze intensity profiles along geometric shape perimeters.")
    
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
                
                # Generate sample coordinates for CSV
                p1 = st.session_state.line_p1
                p2 = st.session_state.line_p2
                num_samples = len(distances)
                x_coords = np.linspace(p1[0], p2[0], num_samples, endpoint=True)
                y_coords = np.linspace(p1[1], p2[1], num_samples, endpoint=True)
                sample_coords = np.array([(int(round(x)), int(round(y))) for x, y in zip(x_coords, y_coords)])
                
                # Convert to units
                pixels_per_unit = st.session_state.get("pixels_per_unit", 1.0)
                units_label = st.session_state.get("scale_units", "pixels")
                distances_units, _ = _convert_to_units(distances, pixels_per_unit)
                
                # Create profile plot
                profile_fig = go.Figure()
                profile_fig.add_trace(go.Scatter(
                    x=distances_units,
                    y=intensities,
                    mode='lines',
                    line=dict(color='cyan', width=2),
                    name='Intensity'
                ))
                xaxis_title = f"Distance ({units_label})" if pixels_per_unit != 1.0 else "Distance (pixels)"
                profile_fig.update_layout(
                    title="Line Profile",
                    xaxis_title=xaxis_title,
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
                
                # CSV Download
                line_points = [p1, p2]
                csv_bytes = _build_profile_csv(
                    line_points, distances, intensities, sample_coords, 
                    pixels_per_unit, units_label
                )
                st.download_button(
                    "📥 Download CSV",
                    data=csv_bytes,
                    file_name="line_profile.csv",
                    mime="text/csv",
                    use_container_width=True,
                    help="Download intensity and coordinate data as CSV"
                )
            
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
                
                # Get perimeter points for CSV
                shape_points = lpt.get_shape_perimeter_points(
                    st.session_state.shape_type,
                    st.session_state.shape_center,
                    st.session_state.shape_size,
                    st.session_state.shape_angle,
                    st.session_state.shape_start_fraction,
                    st.session_state.shape_end_fraction
                )
                
                # Generate sample coordinates (use perimeter points)
                sample_coords = np.array([(int(round(p[0])), int(round(p[1]))) for p in shape_points[:len(distances)]])
                
                # Convert to units
                pixels_per_unit = st.session_state.get("pixels_per_unit", 1.0)
                units_label = st.session_state.get("scale_units", "pixels")
                distances_units, _ = _convert_to_units(distances, pixels_per_unit)
                
                # Create profile plot
                profile_fig = go.Figure()
                profile_fig.add_trace(go.Scatter(
                    x=distances_units,
                    y=intensities,
                    mode='lines',
                    line=dict(color='cyan', width=2),
                    name='Intensity'
                ))
                xaxis_title = f"Distance along perimeter ({units_label})" if pixels_per_unit != 1.0 else "Distance along perimeter (pixels)"
                profile_fig.update_layout(
                    title=f"{st.session_state.shape_type.capitalize()} Perimeter Profile",
                    xaxis_title=xaxis_title,
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
                
                # CSV Download
                csv_bytes = _build_profile_csv(
                    shape_points.tolist(), distances, intensities, sample_coords,
                    pixels_per_unit, units_label
                )
                st.download_button(
                    "📥 Download CSV",
                    data=csv_bytes,
                    file_name=f"{st.session_state.shape_type}_perimeter_profile.csv",
                    mime="text/csv",
                    use_container_width=True,
                    help="Download intensity and coordinate data as CSV"
                )
            else:
                st.info("Configure the line or shape above to see the intensity profile.")

    # Add instructions
    st.markdown("---")
    with st.expander("📖 How to Use", expanded=False):
        st.markdown("""
        **Draw Mode (Default):**
        1. Select draw type: "line" to click two points, "freedraw" to trace any path, or "circle"/"rectangle" to draw shapes
        2. For shapes, click and drag on the canvas to create and resize them
        3. Draw directly on the canvas with your mouse or touch
        4. The intensity profile updates automatically as you draw
        5. Download the CSV to get intensity and coordinate data
        
        **Line Profile Mode:**
        1. Choose "Full Image Line" to create a line spanning the entire image, or uncheck it for a custom line
        2. Use the sliders to adjust the line position, length, and angle
        3. The intensity profile updates automatically as you adjust the controls
        4. For full image lines, only the angle needs to be set
        
        **Shape Profile Mode:**
        1. Select a shape type (circle, triangle, square, pentagon, hexagon, heptagon, or octagon)
        2. Adjust the center position, size, and rotation angle using the sliders
        3. The intensity profile along the perimeter (starting from bottom, going right around) updates automatically
        
        **Note:** Click interactions on the plotly chart are limited in Streamlit. Use the sliders in the sidebar for precise control, or use Draw mode for interactive drawing.
        """)


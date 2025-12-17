import streamlit as st
import numpy as np
import cv2
import plotly.graph_objects as go
import tools.line_profile_tool as lpt

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
        st.session_state.line_profile_mode = "line"  # "line" or "shape"
    
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
    
    # Get image dimensions
    if len(img.shape) == 3:
        height, width = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        height, width = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    # Sidebar controls
    with st.sidebar:
        if st.button("← Back to Workspace", use_container_width=True):
            st.session_state.stage = 4
            st.rerun()
        
        st.markdown("### Mode Selection")
        mode = st.radio(
            "Analysis Mode:",
            ["Line Profile", "Shape Profile"],
            index=0 if st.session_state.line_profile_mode == "line" else 1,
            key="profile_mode_radio"
        )
        st.session_state.line_profile_mode = "line" if mode == "Line Profile" else "shape"
        
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
    
    # Main area: Image with interactive plotly
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


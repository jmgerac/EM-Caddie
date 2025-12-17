"""
Scale Bar UI Module
Handles scale bar configuration interface
"""
import streamlit as st
import cv2
import tools.image_operations as imops


def get_scale_bar_params():
    """Display scale bar configuration UI and return parameters"""
    st.markdown("---")
    st.markdown("### Scale Bar Settings")
    
    with st.container(border=True):
        # Custom units input
        units = st.text_input(
            "Units:",
            value="nm",
            key="scale_bar_units",
            help="Enter the unit for your scale bar (e.g., nm, μm, mm, px)"
        )
        
        # Input mode selection
        input_mode = st.radio(
            "Input Mode:",
            ["Total Image Length", "Pixel to Scale"],
            key="scale_bar_input_mode",
            help="Choose how to specify the scale"
        )
        
        # Get scale parameters based on input mode
        if input_mode == "Total Image Length":
            total_image_length = st.number_input(
                "Total image length:",
                min_value=0.1,
                value=100.0,
                step=1.0,
                key="scale_bar_total_length",
                help=f"The total length of the image in {units}"
            )
            pixel_to_scale = None
        else:  # Pixel to Scale
            pixel_to_scale = st.number_input(
                "Length per pixel:",
                min_value=0.0001,
                value=0.1,
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
            value=10.0,
            step=1.0,
            key="scale_bar_length",
            help=f"The desired length of the scale bar in {units}"
        )
        
        # Position and appearance
        position_param, x_offset, y_offset = _get_position_settings()
        font_scale_factor = _get_font_scale_setting()
        show_background, bar_color, font_color = _get_appearance_settings()
        
        # Store parameters
        scale_bar_params = {
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
        
        # Display preview
        _display_scale_bar_preview(scale_bar_params)
        
        return scale_bar_params


def _get_position_settings():
    """Get position settings for scale bar"""
    position = st.selectbox(
        "Position:",
        ["Bottom Right", "Bottom Left", "Top Right", "Top Left", "Custom"],
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
                value=0.9,
                step=0.01,
                key="scale_bar_x_offset",
                help="0.0 = left edge, 1.0 = right edge"
            )
        
        with col_y:
            y_offset = st.slider(
                "Vertical Position:",
                min_value=0.0,
                max_value=1.0,
                value=0.95,
                step=0.01,
                key="scale_bar_y_offset",
                help="0.0 = top edge, 1.0 = bottom edge"
            )
        
        position_param = "custom"
    else:
        x_offset = None
        y_offset = None
        position_map = {
            "Bottom Right": "bottom_right",
            "Bottom Left": "bottom_left",
            "Top Right": "top_right",
            "Top Left": "top_left"
        }
        position_param = position_map[position]
    
    return position_param, x_offset, y_offset


def _get_font_scale_setting():
    """Get font scale factor setting"""
    return st.slider(
        "Font Size:",
        min_value=0.5,
        max_value=3.0,
        value=1.0,
        step=0.1,
        key="scale_bar_font_scale",
        help="Adjust the font size of the scale bar label"
    )


def _get_appearance_settings():
    """Get appearance settings (background, colors)"""
    st.markdown("---")
    st.markdown("**Appearance Settings:**")
    
    # Background toggle
    show_background = st.checkbox(
        "Show solid black background",
        value=True,
        key="scale_bar_show_background",
        help="Toggle the solid black background box behind the scale bar and text"
    )
    
    # Color pickers
    col_bar, col_font = st.columns(2)
    
    with col_bar:
        bar_color = st.color_picker(
            "Scale Bar Color:",
            value="#FFFFFF",
            key="scale_bar_color",
            help="Choose the color of the scale bar"
        )
    
    with col_font:
        font_color = st.color_picker(
            "Font Color:",
            value="#FFFFFF",
            key="scale_bar_font_color",
            help="Choose the color of the scale bar text"
        )
    
    return show_background, bar_color, font_color


def _display_scale_bar_preview(params):
    """Display preview of image with scale bar"""
    st.markdown("---")
    st.markdown("### Preview")
    
    preview_img = st.session_state.get("working_img")
    if preview_img is not None:
        # Create preview with scale bar
        preview_with_scale = imops.add_scale_bar(
            preview_img.copy(),
            params["scale_bar_length"],
            units=params["units"],
            input_mode=params["input_mode"],
            pixel_to_scale=params["pixel_to_scale"],
            total_image_length=params["total_image_length"],
            position=params["position"],
            x_offset=params["x_offset"],
            y_offset=params["y_offset"],
            font_scale_factor=params["font_scale_factor"],
            show_background=params["show_background"],
            bar_color=params["bar_color"],
            font_color=params["font_color"]
        )
        
        # Convert for display
        if len(preview_with_scale.shape) == 3:
            preview_rgb = cv2.cvtColor(preview_with_scale, cv2.COLOR_BGR2RGB)
        else:
            preview_rgb = preview_with_scale
        
        st.image(preview_rgb, caption="Preview with Scale Bar", use_container_width=True)
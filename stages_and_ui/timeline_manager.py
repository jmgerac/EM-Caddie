"""
Timeline Manager Module
Handles undo/redo timeline functionality
"""
import streamlit as st


def initialize_timeline():
    """Initialize timeline-related session state"""
    if "timeline" not in st.session_state:
        st.session_state.timeline = []
    if "timeline_index" not in st.session_state:
        st.session_state.timeline_index = 0
    if "timeline_images" not in st.session_state:
        st.session_state.timeline_images = [st.session_state.original_img.copy()]


def set_working_from_index():
    """Set working image from timeline index"""
    st.session_state.working_img = (
        st.session_state.timeline_images[st.session_state.timeline_index]
    )


def undo_cb():
    """Undo callback - move back in timeline"""
    if st.session_state.timeline_index > 0:
        st.session_state.timeline_index -= 1
        set_working_from_index()


def redo_cb():
    """Redo callback - move forward in timeline"""
    if st.session_state.timeline_index < len(st.session_state.timeline):
        st.session_state.timeline_index += 1
        set_working_from_index()


def reset_cb():
    """Reset callback - clear timeline and return to original image"""
    orig = st.session_state.original_img.copy()
    st.session_state.timeline = []
    st.session_state.timeline_index = 0
    st.session_state.timeline_images = [orig]
    st.session_state.working_img = orig


def apply_pipeline_cb(tools, scale_bar_params=None):
    """Apply a pipeline of tools to the current image"""
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
            img = _apply_scale_bar(img, scale_bar_params)
        else:
            tool_fn = tools[tool_name][1]
            if tool_fn is not None:
                img = tool_fn(img.copy())
        
        # Record timeline step
        st.session_state.timeline.append(tool_name)
        st.session_state.timeline_images.append(img)
    
    st.session_state.timeline_index = len(st.session_state.timeline)
    st.session_state.working_img = img


def _apply_scale_bar(img, scale_bar_params):
    """Apply scale bar with provided parameters"""
    import tools.image_operations as imops
    
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
    
    return img
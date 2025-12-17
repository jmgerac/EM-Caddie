"""
UI Components Module
Reusable UI components and display functions
"""
import base64
import streamlit as st
import time


def loading_animation():
    """Display loading animation with video"""
    def autoplay_video(path, width=300):
        with open(path, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode()
        html = f"""
            <video autoplay loop muted playsinline style="width:{width}px; height:auto;">
                <source src="data:video/mp4;base64,{b64}" type="video/mp4">
            </video>
            """
        return html
    
    video_html = autoplay_video("assets/quick_golf_miss.mp4", width=300)
    st.markdown(
        f"""
            <div style="display:flex; align-items:center; justify-content:center; gap:50px;">
                <h1 style="font-size:80px; margin:0;">EM Caddie</h1>
                {video_html}
            </div>
            """,
        unsafe_allow_html=True
    )
    time.sleep(0.7)
    st.session_state.stage = 2
    st.rerun()


def display_dimensions_info(width, height, label="Image Dimensions"):
    """Display image dimensions in an info box"""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info(f"**{label}:** {width} × {height} pixels")


def display_metric_columns(width, height):
    """Display width, height, and total pixels in columns"""
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        return st.number_input(
            "Width (px):",
            min_value=1,
            max_value=width,
            value=int(width),
            step=1,
            key="crop_width_input",
            help="Click to input the desired width in pixels"
        )
    with col2:
        return st.number_input(
            "Height (px):",
            min_value=1,
            max_value=height,
            value=int(height),
            step=1,
            key="crop_height_input",
            help="Click to input the desired height in pixels"
        )
    with col3:
        total_pixels = width * height
        st.metric("Total Pixels", f"{total_pixels:,}")
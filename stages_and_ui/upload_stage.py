"""
Upload Stage Module
Handles initial image upload functionality
"""
import streamlit as st
import numpy as np
import cv2


def initial_upload():
    """Stage for initial image upload"""
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
        st.session_state.working_img = cv2_img.copy()
        st.session_state.stage = 3  # Go to cropping stage
        st.rerun()
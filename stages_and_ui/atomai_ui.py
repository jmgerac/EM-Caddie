"""
Segmentation Output UI Module
Handles probability vs binary threshold selection
"""

import streamlit as st
import cv2
import numpy as np


def segmentation_output_controls(pred_mask_uint8):
    """
    UI for choosing segmentation output type and applying it.

    Parameters
    ----------
    pred_mask_uint8 : np.ndarray
        Probability mask in uint8 [0,255]

    Returns
    -------
    np.ndarray or None
        Processed mask when Apply is pressed, else None
    """

    st.markdown("---")
    st.markdown("### Segmentation Output")

    with st.container(border=True):

        output_mode = st.radio(
            "Output type:",
            ["Probability mask", "Binary threshold"],
            index=0,
            key="seg_output_mode"
        )

        threshold = None
        if output_mode == "Binary threshold":
            threshold = st.slider(
                "Threshold",
                min_value=0,
                max_value=255,
                value=128,
                step=1,
                key="seg_threshold"
            )

        apply = st.button("Apply segmentation", type="primary")

        if not apply:
            return None

        # ---- Apply logic ----
        if output_mode == "Probability mask":
            return pred_mask_uint8.copy()

        # Binary threshold
        _, binary = cv2.threshold(
            pred_mask_uint8,
            threshold,
            255,
            cv2.THRESH_BINARY
        )
        return binary

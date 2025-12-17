"""
Main Application File (Refactored)
Orchestrates the different stages of the application
"""
import streamlit as st
from stages_and_ui.ui_components import loading_animation
from stages_and_ui.upload_stage import initial_upload
from stages_and_ui.crop_stage import image_cropper
from stages_and_ui.workspace_ui import workspace


def main(tools, tool_names, tool_embs, encoder):
    """
    Main application entry point
    
    Args:
        tools: Dictionary of available tools
        tool_names: List of tool names
        tool_embs: Tool embeddings for similarity matching
        encoder: Encoder for query interpretation
    """
    # Initialize stage if not present
    if "stage" not in st.session_state:
        st.session_state.stage = 1
    
    # Route to appropriate stage
    stage = st.session_state.stage
    
    if stage == 1:
        loading_animation()
    elif stage == 2:
        initial_upload()
    elif stage == 3:
        image_cropper()
    elif stage == 4:
        workspace(tools, tool_names, tool_embs, encoder)
    elif stage == 5:
        # Line profile tool (handled elsewhere)
        import stages_and_ui.line_profile_ui as line_profile_ui
        # Call line profile UI here if needed
        pass
    else:
        st.error(f"Unknown stage: {stage}")


# Example usage:
# if __name__ == "__main__":
#     from app_context import get_tools, get_encoder
#     tools, tool_names, tool_embs = get_tools()
#     encoder = get_encoder()
#     main(tools, tool_names, tool_embs, encoder)

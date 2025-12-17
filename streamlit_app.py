import streamlit as st
# Set page config FIRST before any other Streamlit commands
st.set_page_config(layout="wide")

import stages
import app_context
import stages_and_ui.line_profile_ui as line_profile_ui
# Initialize global resources and formatting
encoder = app_context.get_encoder()
tools, tool_names, tool_descs, tool_embs = app_context.get_tools()

# app_context.set_text_formatting()

# Session state
st.session_state.setdefault("stage", 1)
st.session_state.setdefault("original_img", None)
st.session_state.setdefault("working_img", None)

# Stage routing with arguments
stage_actions = {
    1: (stages.loading_animation, ()),
    2: (stages.initial_upload, ()),
    3: (stages.image_cropper, ()),
    4: (stages.workspace, (tools, tool_names, tool_embs, encoder)),
    5: (line_profile_ui.line_profile_stage, ()),
}

# Run stage
stage_fn, args = stage_actions.get(st.session_state.stage, (None, ()))
if stage_fn:
    stage_fn(*args)

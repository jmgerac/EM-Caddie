import os
import streamlit as st
import atomai
from sentence_transformers import SentenceTransformer
import tools.image_operations as imops
import torch
import redivis
import sys
import TEMUpSamplerNet_app.model as tem_model
from TEMUpSamplerNet_app.model import Net
sys.modules["model"] = tem_model

tool_aliases = {
    "fft": "Fast Fourier Transform (FFT)",
    "fourier": "Fast Fourier Transform (FFT)",
    # Add more as needed
}

def resolve_tool_name(name: str) -> str:
    name = name.strip().lower()
    return tool_aliases.get(name, name)


from unet import UNet

from unet_model.inference.infer import single_inference
from utility.settings import MODEL_PARAMS


@st.cache_resource
def load_encoder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_segmentor():
    gmd_tar_path = r"atomai_app/pretrained/G_MD.tar"
    return atomai.load_model(gmd_tar_path)

@st.cache_resource
def load_super_res_model(device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path = r"TEMUpSamplerNet_app/denoise&bgremoval2x.pth"

    # Use safe_globals to allow the Net class
    with torch.serialization.safe_globals([Net]):
        net = torch.load(path, map_location=device, weights_only=False)

    net.to(device)
    net.eval()
    return net

@st.cache_resource
def load_grain_unet(device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights_path = r"grain_unet-master/grain_unet-master/grain_unet_pyotrch_100epoch.pth"
    # if weights_path doesn't exist
    if not os.path.exists(weights_path):
        file = redivis.file("wtgg-5c9t33qqt.lvIUsKbKP5h5mBIMtn2LCQ")
        file.download(weights_path)
    model = UNet(out_channels=1)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    return model

@st.cache_resource
def set_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def prep_tools():
    tools = {
        "Edge Detection": ("Detect edges in an image using the Canny algorithm.", imops.edge_detect),
        "Gaussian Blur": ("Blur the image using a Gaussian kernel.", imops.blur),
        "Invert Colors": ("Invert all the colors in the image.", imops.invert),
        "Fast Fourier Transform (FFT)": ("Take the fast fourier transform of the image.", imops.fft),
        "Super Resolution": ("Denoise and upsample the image to create super resolution.", lambda img: imops.super_res(img, super_res_model, device)),
        "AtomAI Segmentation": ("Identify and highlight atom centers.",
                                lambda img: imops.atomai_segment(img, segmentor)),
        "Grain Boundary Probability Map": ("Estimate grain boundaries as a confidence map.", lambda img: imops.grain_unet_segment(img, grain_model, device)),
        "Add Scale Bar": ("Add a scale bar to the bottom right corner of the image.", None),
    }
    tool_names = list(tools.keys())
    tool_descs = [desc for desc, fn in tools.values()]
    tool_embs = encoder.encode(tool_descs, normalize_embeddings=True)

    return tools, tool_names, tool_descs, tool_embs

encoder = load_encoder()
segmentor = load_segmentor()
super_res_model = load_super_res_model()
grain_model = load_grain_unet()
device = set_device()

tools, tool_names, tool_descs, tool_embs = prep_tools()

def interpret_query(
    query: str, threshold: float = 0.5) -> list[str]:
    """
    Interpret the user's query and return relev ant tool names.
    Splits on newline, comma, and 'and'.
    """
    if not query:
        return []

    query = query.lower().replace(",", "\n").replace("and", "\n")
    query_parts = [q.strip() for q in query.split("\n") if q.strip()]

    resolved_tools = []

    # Apply aliases
    remaining_parts = []
    for part in query_parts:
        alias_hit = tool_aliases.get(part)
        if alias_hit is not None:
            resolved_tools.append(alias_hit)
        else:
            remaining_parts.append(part)

    if not remaining_parts:
        return resolved_tools

    # Encode all query parts at once
    query_embs = encoder.encode(
        query_parts,
        normalize_embeddings=True
    )  # shape: (M, D)

    # Similarity matrix: (M, D) @ (D, N) -> (M, N)
    scores = query_embs @ tool_embs.T

    for row in scores:
        best_idx = row.argmax()
        if row[best_idx] >= threshold:
            resolved_tools.append(tool_names[best_idx])
        else:
            resolved_tools.append("")

    return resolved_tools

def basic_tool_component(tool_name: str, tools: dict):
    """
    Builds a small component that pops up when a user selects a basic tool.
    It shows the name and description of the tool and has a single button, "Apply"

    :param tool_name: The name of the selected tool (e.g., "Edge Detection").
    :param tools: The global dictionary containing tool details (desc, fn).
    :return: True if the "Apply" button was clicked, False otherwise.
    """
    # Get the description from the global tools dictionary
    canonical_name = resolve_tool_name(tool_name)
    tool_desc = tools[canonical_name][0]

    st.markdown("## **Suggested Tool:**",
                unsafe_allow_html=False)

    # Use a container to visually group the component
    with st.container(border=True):
        # Display Tool Name (Larger/Bold)
        st.markdown(f"## **{tool_name}**",
                    unsafe_allow_html=False)

        # Display Tool Description
        st.markdown(f"{tool_desc}",
                    unsafe_allow_html=False)

        st.markdown("---")  # Visual separator

        # Create the 'Apply' button
        # Use a single &nbsp; to allow fit-to-content width, or add more for padding
        apply_clicked = create_button(
            "Apply Tool",
            key=f"apply_{tool_name.replace(' ', '_')}",
            # Note: No on_click is needed here, as the action is handled by workspace
        )

    return apply_clicked


def display_tool_help():
    """
    Displays a help message in the area where the tool component would appear.
    """
    # Use st.container(border=True) or st.info to match the visual weight of the tool component
    with st.container(border=True):
        st.markdown(
            "## No tool suggested.",
            unsafe_allow_html=False
        )
        st.markdown(
            """
            Try rephrasing your request or use the sidebar dropdown to select a tool manually.
            """,
            unsafe_allow_html=False
        )
        st.markdown("---")
        # Optional: Suggest general actions or examples
        st.markdown(
            "**Examples:** 'Find the edges,' 'Blur the noise,' or 'Invert the image colors.'"
        )

    # The help message does not enable applying a tool
    return False

def create_button(label: str, key: None | str = None, on_click=None, args=None, kwargs=None, type='primary', icon=None, disabled=False,):
    # Changed to default button for stability
    return st.button(
    label=label,
    key=key,
    on_click=on_click,
    args=args,
    kwargs=kwargs,
    use_container_width=True,
    type=type, #type: ignore 
    disabled=disabled,
    icon=icon
    )
    # return st.button(label, key=key, on_click=on_click, args=args, kwargs=kwargs)


def set_layout():
    st.set_page_config(layout="wide")


def get_encoder():
    return encoder

def get_segmentor():
    return segmentor

def get_tools():
    return tools, tool_names, tool_descs, tool_embs
# EM-Caddie

![Logo GIF](https://github.com/jmgerac/EM-Caddie/blob/3be6addfb05d2fc0ccf92490bdb4b01c62faaac4/assets/good_white_background.gif)

**Intelligent Microscopy Image Analysis - No Coding Required**

EM-Caddie is a web-based tool that brings the power of AI to microscopy researchers. Simply describe what you want to do in plain English, upload your images, and let our integrated suite of pre-trained models handle the analysis.

*Developed for the [Microscopy Hackathon, 2025](https://kaliningroup.github.io/mic_hackathon_2/about/)*
**Disclaimer:** Some initial brainstorming and preliminary development, including a prototype interface, for this project occurred before the hackathon. Almost all feature addition, development, and testing ocurred during the Hackathon, Dec 16-18 2025.


![Example GIF](https://github.com/jmgerac/EM-Caddie/blob/da6ffbae2153c765abe58ba3bbb0e152af3eb54e/assets/video-animation.gif)

## What Does EM-Caddie Do?

- **Natural Language Interface**: Tell the system what you need in plain English (e.g., "enhance the resolution and detect edges")
- **Automated Image Processing**: Access state-of-the-art pre-trained models for enhancement, segmentation, and analysis
- **Interactive Analysis Tools**: Explore your microscopy data through an intuitive web interface
- **No Programming Required**: Everything works through a user-friendly web application

Perfect for electron microscopy, atomic force microscopy, and other high-resolution imaging techniques.

## Capabilities

### Image Enhancement & Processing
- **Super Resolution**: AI-powered upscaling to reveal finer details in your images
- **Gaussian Blur**: Smooth noise while preserving important features
- **Edge Detection**: Automatically identify boundaries and structures
- **Invert Colors**: Quick contrast adjustments for better visualization
- **Add Scale Bar**: Easy annotations with customizable scale markers

### Advanced Analysis
- **AtomAI Segmentation**: Deep learning-based segmentation optimized for atomic-scale imaging
- **Fast Fourier Transform (FFT)**: Frequency domain analysis for crystallographic studies
- **Line Profile Analysis**: Measure intensity variations along custom lines or shapes
- **Interactive Cropping**: Select and extract regions of interest with precision

### Smart Features
- **Conversational Interface**: Chain multiple operations together using natural language
- **Visual Feedback**: Real-time preview of all processing steps
- **Edit Timeline**: Easily undo/redo image operations 

## Installation

### Prerequisites

You'll need [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on your computer.
**Note**: If you have an NVIDIA GPU, the software will automatically use it for faster processing. If not, it will work fine on your CPU (just a bit slower).

### Installation

1. **Download this project** to your computer
   ```bash
   git clone https://github.com/jmge
   ```
3. **Open a terminal** (Command Prompt on Windows, Terminal on Mac/Linux)
   ```bash
   # Create the environment
   conda env create -f environment.yml
   
   # Activate the environment
   conda activate em-caddie
   ```
That's it! The installation takes a few minutes, so take a little break!
On the first run, a popup will prompt you to sign in to a [Redivis](https://redivis.com/) account, in order to access some of our models.

## Running EM-Caddie

Once installed, follow these simple steps each time you want to use the application:

1. **Open a terminal** (Command Prompt on Windows, Terminal on Mac/Linux)

2. **Activate the environment**:
   ```bash
   conda activate em-caddie
   ```

3. **Start the application**:
   ```bash
   streamlit run streamlit_app.py
   ```

4. **Open your browser** - The application will automatically open at http://localhost:8501

5. **When you're done**, press `Ctrl+C` in the terminal to stop the application

## Using EM-Caddie     

1. **Upload your microscopy images** through the web interface (supports common formats: TIFF, PNG, JPEG)
2. **Prepare your image** using the crop tool
3. **Try the natural language interface**: Type commands like:
   - "Apply super resolution and then detect edges"
   - "Show me the FFT of this image"
   - "Segment the atomic structures and add a scale bar"
4. **Analyze and export**: View results, compare before/after, and download processed images

![Line Profile Tool Example GIF](https://github.com/jmgerac/EM-Caddie/blob/d636b1beaa2d73adf728a065a4e8ce70f9229d59/assets/lineprofile.gif)

### Example Pipelines

**Enhancing Image Quality:**
- Upload → Super Resolution → Gaussian Blur → Add Scale Bar

**Structural Analysis:**
- Upload → Crop Region → AtomAI Segmentation → Line Profile Analysis

**Frequency Analysis:**
- Upload → Edge Detection → FFT → Export Results

No programming knowledge needed - just describe what you want or click through the interface!

## Troubleshooting

### "The application won't start"
- Make sure you've activated the environment first: `conda activate em-caddie`
- Check that you're in the correct folder (where `streamlit_app.py` is located)

### "Installation is taking forever"
- This is normal! The first installation downloads several gigabytes of AI models
- Grab a coffee - it usually takes a few minutes, depending on your internet connection

## About This Project

EM-Caddie was developed for the [Electron Microscopy Foundation Hackathon 2025](https://kaliningroup.github.io/mic_hackathon_2/about/)

Traditional microscopy image analysis requires:
- Learning complex software packages
- Writing custom scripts
- Understanding machine learning frameworks
- Significant time investment

**EM-Caddie changes that.** By integrating multiple state-of-the-art pre-trained models into a single web interface with natural language understanding, we enable researchers to focus on science, not software.

### Technology Stack
- **AtomAI**: Deep learning for atomic-scale image analysis
- **PyTorch**: Neural network framework powering super resolution
- **Sentence Transformers**: Natural language understanding
- **Streamlit**: Interactive web interface

**No model training required - just analysis.**

---

## Roadmap

This roadmap lays out a set of prioritized and realistic next steps aimed at carrying EM-Caddie beyond its hackathon origins and toward a mature, extensible microscopy analysis platform. While the current version already demonstrates what’s possible when modern AI tools are put directly in researchers’ hands, the goal moving forward is to turn that proof-of-concept into something stable, scalable, and genuinely useful in day-to-day scientific workflows—without losing the speed, creativity, and experimental spirit that made the hackathon build possible in the first place.

### Short-Term (Stability & Usability)

**Improve Reliability of Existing Tools**
- Fix known issues with the line drawing / intensity extraction tool
- Add clearer user feedback when tools return invalid or partial results

**Memory-Aware Execution**
- Estimate memory usage before running computationally intensive models
- Warn users when an image or model may exceed available system resources
- Improve cache management for large images and intermediate results

**Standardized Model "Plugin" Format**
- Adding additional features and models becomes easier if there is a simple, consistent way to define how a model is loaded, what inputs it expects, and what outputs it produces.
- A lightweight plugin structure would reduce the need to modify core code when experimenting with new models, which is especially valuable in a fast-paced hackathon setting.
- This also helps keep the codebase organized as the number of supported tools grows.


### Mid-Term (Expanded Analysis Capabilities)

**Automatic Scale Bar Detection & Calibration**
- Detect embedded scale bars in micrographs
- Extract scale text (e.g., nm, Å) and auto-calibrate pixel dimensions
- Enable consistent, reproducible quantitative measurements

**Grain Counting & Density Analysis**
- Manual and semi-automatic grain annotation tools
- Grain size distributions, areal density, and spacing statistics
- Export results as CSV and publication-ready figures

**Expanded Model Library**
- Add additional pre-trained models for segmentation, enhancement, and measurement
- Introduce a standardized plugin interface for new tools
- Enable community contributions without modifying core code


### Long-Term (Intelligent Assistance & Automation)

**ML-Based Image Identification**
- Automatically infer microscopy modality (TEM, SEM, AFM, etc.)
- Detect common structural features (grains, lattices, defects)
- Use predictions to suggest appropriate analysis tools

**Image-Based Pipeline Suggestions**
- Recommend complete analysis workflows based on image content
- One-click execution with editable pipeline previews
- Rank suggestions by relevance and computational cost

**LLM-Assisted Scientific Guidance**
- Explain analysis steps and results in context
- Help users interpret FFTs, line profiles, and segmentation outputs
- Provide guidance without requiring ML or image-processing expertise

---

Together, these roadmap items focus on increasing robustness first, expanding quantitative capabilities next, and ultimately enabling EM-Caddie to act as an intelligent microscopy analysis assistant rather than just a collection of tools.

# EM-Caddie

**Intelligent Microscopy Image Analysis - No Coding Required**

EM-Caddie is a web-based tool that brings the power of AI to microscopy researchers. Simply describe what you want to do in plain English, upload your images, and let our integrated suite of pre-trained models handle the analysis.

*Developed for the [Electron Microscopy Foundation Hackathon 2025](https://kaliningroup.github.io/mic_hackathon_2/about/)*

![Example GIF](https://github.com/jmgerac/EM-Caddie/blob/cc33805ad0cb093b38d48f4b70ef2b12f9aa1644/assets/chat.gif?raw=true)

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
- **Add Scale Bar**: Professional annotations with customizable scale markers

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

You'll need [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on your computer. Don't worry - this is a one-time setup that makes installing scientific software much easier.

**Note**: If you have an NVIDIA GPU, the software will automatically use it for faster processing. If not, it will work fine on your CPU (just a bit slower).

### Easy Installation (Recommended)

1. **Download this project** to your computer
2. **Open a terminal** (Command Prompt on Windows, Terminal on Mac/Linux)
3. **Navigate to the project folder**:
   ```bash
   cd path/to/EM-Caddie
   ```
4. **Run the installer**:
   ```bash
   chmod +x install.sh
   ./install.sh
   ```

The installer will:
- Set up everything automatically
- Test that it's working
- Tell you what to do next

That's it! The installation takes a few minutes, so take a little break!

### Alternative: Manual Installation

If the automatic installer doesn't work, you can install manually:

```bash
# Create the environment
conda env create -f environment.yml

# Activate the environment
conda activate em-caddie
```

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

### Example Workflows

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

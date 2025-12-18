
import configparser
import torch
import torch.nn as nn
from pathlib import Path
global INFERENCE, IMAGE_PATH, OUTPUT_PATH, DEVICE_COMPUTE_PLATFORM, LABEL_PATH, BATCH_SIZE, NUM_EPOCHS, NUM_WORKERS
global MODEL_PARAMS, MODEL_NAME, TEST_DATA_DIR, PREFIX, PP_ACTIVATE
global PREDICT_DATA_DIR, TARGET_RESOLUTION, FINAL_OUTPUT_DIRECTORY, args_pp
global TRAINING_SPLIT, AUGMENT_ACTIVE, LEARNING_RATE
global CUSTOM_LOSS_FUNCTION, UNET_MODEL, LOSS_FN, OPTIMIZER, SAVING_RATE
# Expected input size for grain UNet inference
TARGET_RESOLUTION = 512
DEVICE_COMPUTE_PLATFORM = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)
MODEL_PARAMS = str(
    Path(__file__).resolve().parent.parent
    / "grain_unet-master"
    / "grain_unet-master"
    / "grain_unet_pyotrch_100epoch.pth"
)
PREFIX = "pred_"
def init(config_file_name: str = "MyUnetConfig.ini") -> None:
    """
    Initialize settings.
    Safe for inference-only usage when training config sections are missing.
    """

    global INFERENCE, IMAGE_PATH, OUTPUT_PATH, DEVICE_COMPUTE_PLATFORM, LABEL_PATH
    global BATCH_SIZE, NUM_EPOCHS, NUM_WORKERS, MODEL_PARAMS, MODEL_NAME
    global TEST_DATA_DIR, PREFIX, PP_ACTIVATE, PREDICT_DATA_DIR
    global TARGET_RESOLUTION, FINAL_OUTPUT_DIRECTORY, args_pp
    global TRAINING_SPLIT, AUGMENT_ACTIVE, LEARNING_RATE, N_TEST_PATTERN

    config = configparser.ConfigParser()
    config_path = Path(__file__).resolve().parents[1] / config_file_name
    config.read(config_path)

    # -----------------------------
    # Device (always valid)
    # -----------------------------
    DEVICE_COMPUTE_PLATFORM = get_compute_platform()

    # -----------------------------
    # DATA PARAMS (training only)
    # -----------------------------
    if 'DATA_PARAMS' in config:
        AUGMENT_ACTIVE = int(config['DATA_PARAMS'].get('AUGMENT_ACTIVE', 0))
        TRAINING_SPLIT = config['DATA_PARAMS'].get('TRAINING_SPLIT', None)
    else:
        AUGMENT_ACTIVE = 0
        TRAINING_SPLIT = None

    # -----------------------------
    # PATHS (optional for inference)
    # -----------------------------
    if 'PATH' in config:
        IMAGE_PATH = config['PATH'].get('IMAGE_PATH', None)
        LABEL_PATH = config['PATH'].get('LABEL_PATH', None)
        MODEL_PARAMS = config['PATH'].get('MODEL_PARAMS', MODEL_PARAMS)
    else:
        IMAGE_PATH = None
        LABEL_PATH = None
        # MODEL_PARAMS already defined at module level

    MODEL_NAME = Path(MODEL_PARAMS).stem

    # -----------------------------
    # TRAINING PARAMS (optional)
    # -----------------------------
    if 'TRAINING_PARAMS' in config:
        BATCH_SIZE = int(config['TRAINING_PARAMS'].get('BATCH_SIZE', 1))
        NUM_EPOCHS = int(config['TRAINING_PARAMS'].get('EPOCHS', 0))
        NUM_WORKERS = int(config['TRAINING_PARAMS'].get('NUM_WORKERS', 0))
        LEARNING_RATE = float(config['TRAINING_PARAMS'].get('LEARNING_RATE', 0.0))
    else:
        BATCH_SIZE = 1
        NUM_EPOCHS = 0
        NUM_WORKERS = 0
        LEARNING_RATE = 0.0

    # -----------------------------
    # INFERENCE PARAMS
    # -----------------------------
    if 'INFERENCE_PARAMS' in config:
        TARGET_RESOLUTION = int(config['INFERENCE_PARAMS'].get('OUTPUT_RESOLUTION', TARGET_RESOLUTION))
        INFERENCE = config['INFERENCE_PARAMS'].get('INFERENCE', True)
        TEST_DATA_DIR = config['INFERENCE_PARAMS'].get('N_TEST_DATA_DIR', None)
        N_TEST_PATTERN = config['INFERENCE_PARAMS'].get('N_TEST_PATTERN', None)
        PREDICT_DATA_DIR = config['INFERENCE_PARAMS'].get('OVERLAY_DATA_DIR', None)
        FINAL_OUTPUT_DIRECTORY = config['INFERENCE_PARAMS'].get('FINAL_OUTPUT_DIR', None)
    else:
        INFERENCE = True
        TEST_DATA_DIR = None
        N_TEST_PATTERN = None
        PREDICT_DATA_DIR = None
        FINAL_OUTPUT_DIRECTORY = None

    # -----------------------------
    # POST PROCESS (optional)
    # -----------------------------
    if 'POST_PROCESS' in config:
        PP_ACTIVATE = int(config['POST_PROCESS'].get('POST_PROCESS', 0))
        args_pp = {
            'compilation': config['POST_PROCESS'].get('COMPILATION', None),
            'n_dilations': int(config['POST_PROCESS'].get('N_DILATIONS', 0)),
            'liberal_thresh': int(config['POST_PROCESS'].get('LIBERAL_THRESHOLD', 0)),
            'conservative_thresh': int(config['POST_PROCESS'].get('CONSERVATIVE_THRESHOLD', 0)),
            'invert_double_thresh': bool(config['POST_PROCESS'].get('INVERT_DOUBLE_THRESHOLD', False)),
            'min_grain_area': int(config['POST_PROCESS'].get('MIN_GRAIN_AREA', 0)),
            'prune_size': int(config['POST_PROCESS'].get('PRUNE_SIZE', 0)),
        }
    else:
        PP_ACTIVATE = 0
        args_pp = {}



def init_training(config_file_name:str = "MyUnetConfig.ini")->None :
    
    global CUSTOM_LOSS_FUNCTION, UNET_MODEL, LOSS_FN, DEVICE_COMPUTE_PLATFORM, OPTIMIZER, SAVING_RATE
    init()

    config = configparser.ConfigParser()
    config.read(config_file_name)
    pretrained_weights = config['TRAINING_PARAMS']['PRETRAINED_WEIGHTS']
    CUSTOM_LOSS_FUNCTION = config['TRAINING_PARAMS']['CUSTOM_LOSS_FUNCTION']
    SAVING_RATE = int(config['TRAINING_PARAMS']['SAVING_RATE'])

    try:
        num_class = int(config['TRAINING_PARAMS']['NUM_CLASSES'])
    except:
        num_class = 1

    if CUSTOM_LOSS_FUNCTION:
        LOSS_FN = CUSTOM_LOSS_FUNCTION
    else:
        LOSS_FN = nn.BCEWithLogitsLoss()

    if pretrained_weights:
        UNET_MODEL = pretrained_weights
    else:
        UNET_MODEL = None

    OPTIMIZER = torch.optim.Adam
    if CUSTOM_LOSS_FUNCTION == "BCEWithLogitsLoss":
        LOSS_FN = nn.BCEWithLogitsLoss()
    else:
        Warning("No loss function specified. Using default BCEWithLogitsLoss")
        LOSS_FN = nn.BCEWithLogitsLoss()

def get_compute_platform():

    if torch.cuda.is_available():
        compute_platform = 'cuda'
    elif torch.backends.mps.is_available():
        compute_platform = 'mps'
    else:
        compute_platform = 'cpu'
    return compute_platform

# DO NOT auto-run init on import
# init() and init_training() must be called explicitly



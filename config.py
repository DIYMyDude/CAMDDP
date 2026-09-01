#========================= config.py =========================
#This file contains the application wide settings for CAMDDP.
#=============================================================

from pathlib import Path
import torch

# =========================
# PROJECT PATHS
# =========================
DATA_DIR = Path("E:/data_AVDEEPFAKE1M_10000")        # Processed data (features, outputs)

# =========================
# AUDIO SETTINGS
# =========================

AUDIO_FILTERS = [
    32,
    64,
    128
]

AUDIO_KERNEL_SIZE = 3

AUDIO_USE_BATCHNORM = True

AUDIO_ACTIVATION = "relu"

AUDIO_POOL_TYPE = "max"

AUDIO_POOL_SIZE = 2

AUDIO_USE_GAUSSIAN_NOISE = True

AUDIO_GAUSSIAN_NOISE_STD = 0.05

AUDIO_USE_GAP = True

AUDIO_EMBEDDING_DIM = 128 #128

AUDIO_USE_DROPOUT = False

AUDIO_DROPOUT_RATE = 0.0

AUDIO_MFCC_COEFFICIENTS = 13 #13

# =========================
# VISUAL SETTINGS
# =========================

VISUAL_FEATURE_DIM = 1280

VISUAL_USE_FRAME_DIFFERENCE = True


# =========================
# COMMON SETTINGS
# =========================
CATEGORY_MAP = {
    "A": 0,   # real
    "B": 1,   # fake
    "C": 1,   # fake
    "D": 1    # fake
}

CATEGORY_4_MAP = {
    "A": 0,   # audio real, visual real
    "B": 1,   # audio fake, visual real
    "C": 2,   # audio real, visual fake
    "D": 3    # audio fake, visual fake
}

VISUAL_CATEGORY_MAP = {
    "A": 0,   # visual real
    "B": 0,   # visual real
    "C": 1,   # visual fake
    "D": 1    # visual fake
}

AUDIO_CATEGORY_MAP = {
    "A": 0,
    "B": 1,
    "C": 0,
    "D": 1
}

BATCH_SIZE = 8

NUM_WORKERS = 4

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

STAGE1_EPOCHS = 80

STAGE2_EPOCHS = 25

STAGE1_LR = 1e-3

STAGE2_LR = 1e-5

EARLY_STOPPING_PATIENCE = 10

BEST_MODEL_METRIC = "f1"

FUSION_USE_HIDDEN_LAYER = True

FUSION_HIDDEN_DIM = 256

#----TRAIN----
CHECKPOINT_DIR = Path(
    "checkpoints"
)

CHECKPOINT_DIR = Path(
    "checkpoints"
)

STAGE1_BEST_FILE = (
    CHECKPOINT_DIR /
    "stage1_best.pth"
)

STAGE2_BEST_FILE = (
    CHECKPOINT_DIR /
    "stage2_best.pth"
)

BEST_MODEL_FILE = (
    CHECKPOINT_DIR /
    "best_model.pth"
)

LATEST_MODEL_FILE = (
    CHECKPOINT_DIR /
    "latest.pth"
)

HISTORY_FILE = (
    CHECKPOINT_DIR /
    "training_history.csv"
)

TRAIN_LOG_DIR = (
    Path("logs") /
    "training"
)

STAGE1_4_BEST_FILE = (
    CHECKPOINT_DIR /
    "stage1_best_4.pth"
)

STAGE2_4_BEST_FILE = (
    CHECKPOINT_DIR /
    "stage2_best_4.pth"
)

BEST_4_MODEL_FILE = (
    CHECKPOINT_DIR /
    "best_model_4.pth"
)

LATEST_4_MODEL_FILE = (
    CHECKPOINT_DIR /
    "latest_4.pth"
)

HISTORY_4_FILE = (
    CHECKPOINT_DIR /
    "training_history_4.csv"
)


#----TEST----
TEST_RESULTS_DIR = Path(
    "results"
)

TEST_RESULTS_DIR = Path(
    "results"
)

TEST_PREDICTIONS_FILE = (
    TEST_RESULTS_DIR /
    "test_predictions.csv"
)

TEST_CONFUSION_MATRIX_FILE = (
    TEST_RESULTS_DIR /
    "test_confusion_matrix.csv"
)

TEST_REPORT_FILE = (
    TEST_RESULTS_DIR /
    "test_report.txt"
)

TEST_LOG_DIR = (
    Path("logs") /
    "testing"
)

TEST_4_PREDICTIONS_FILE = (
    TEST_RESULTS_DIR /
    "test_predictions_4.csv"
)

TEST_4_CONFUSION_MATRIX_FILE = (
    TEST_RESULTS_DIR /
    "test_confusion_matrix_4.csv"
)

TEST_4_REPORT_FILE = (
    TEST_RESULTS_DIR /
    "test_report_4.txt"
)

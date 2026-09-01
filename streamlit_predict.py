#========================= streamlit_predict.py =========================
#This file contains the functions that load pre-tained models and perform inference in the Streamlit main web application of CAMDDP.
#========================================================================

from pathlib import Path
import time

import torch

from model_4class import Model

from config import DEVICE

from streamlit_audio_inference import (
    preprocess_audio
)

from streamlit_visual_inference import (
    preprocess_visual
)

# ==================================================
# MODEL CONFIGURATION
# ==================================================

MODEL_CONFIGS = {

    "CAMDDP with FakeAVCeleb Trained Model": {
        "checkpoint": Path(
            "streamlit_checkpoints/FakeAVCeleb/camddp.pth"
        ),
        "vdiff": False
    },

    "CAMDDP (VDIFF) with FakeAVCeleb Trained Model": {
        "checkpoint": Path(
            "streamlit_checkpoints/FakeAVCeleb/camddp_vdiff.pth"
        ),
        "vdiff": True
    },

    "CAMDDP with LAV-DF Trained Model": {
        "checkpoint": Path(
            "streamlit_checkpoints/LAVDF/camddp.pth"
        ),
        "vdiff": False
    },

    "CAMDDP (VDIFF) with LAV-DF Trained Model": {
        "checkpoint": Path(
            "streamlit_checkpoints/LAVDF/camddp_vdiff.pth"
        ),
        "vdiff": True
    }
}

# ==================================================
# CLASS MAPPING
# ==================================================

CLASS_MAP = {

    0: (
        "A",
        "Real Audio + Real Visual"
    ),

    1: (
        "B",
        "Fake Audio + Real Visual"
    ),

    2: (
        "C",
        "Real Audio + Fake Visual"
    ),

    3: (
        "D",
        "Fake Audio + Fake Visual"
    )
}

# ==================================================
# LOAD MODEL
# ==================================================

def load_model(
    model_name
):

    if model_name not in MODEL_CONFIGS:

        raise ValueError(
            f"Unknown model: {model_name}"
        )

    checkpoint = MODEL_CONFIGS[
        model_name
    ]["checkpoint"]

    if not checkpoint.exists():

        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}"
        )

    model = Model()

    model.load_state_dict(
        torch.load(
            checkpoint,
            map_location=DEVICE
        )
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    return model

# ==================================================
# PREDICT VIDEO
# ==================================================

def predict_video(
    video_path,
    model,
    use_frame_difference=False
):

    start_time = time.time()

    # ==============================================
    # AUDIO PREPROCESSING
    # ==============================================

    mel = preprocess_audio(
        video_path
    )

    mel = mel.to(
        DEVICE
    )

    # ==============================================
    # VISUAL PREPROCESSING
    # ==============================================

    frames = preprocess_visual(
        video_path,
        use_frame_difference=use_frame_difference
    )

    frames = frames.to(
        DEVICE
    )

    # ==============================================
    # MODEL INFERENCE
    # ==============================================

    with torch.no_grad():

        outputs, _, _, pa, pv = model(
            mel,
            frames,
            return_features=True
        )

        probs = torch.softmax(
            outputs,
            dim=1
        )

    # ==============================================
    # PREDICTION
    # ==============================================

    pred_idx = torch.argmax(
        probs,
        dim=1
    ).item()

    prediction, description = (
        CLASS_MAP[
            pred_idx
        ]
    )

    probs = probs.cpu().numpy()[0]

    runtime = (
        time.time()
        - start_time
    )

    # ==============================================
    # RETURN RESULTS
    # ==============================================

    return {

        "prediction":
            prediction,

        "description":
            description,

        "pa":
            float(
                pa.item()
            ),

        "pv":
            float(
                pv.item()
            ),

        "prob_a":
            float(
                probs[0]
            ),

        "prob_b":
            float(
                probs[1]
            ),

        "prob_c":
            float(
                probs[2]
            ),

        "prob_d":
            float(
                probs[3]
            ),

        "inference_time":
            float(
                runtime
            )
    }
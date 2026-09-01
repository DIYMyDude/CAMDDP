#========================= streamlit_audio_inference.py =========================
#This file contains the supporting audio preprocessing of uploaded samples for the Streamlit inference/test web application of CAMDDP application.
#================================================================================

import os
import tempfile
import subprocess

import librosa
import numpy as np
import soundfile as sf
import torch

# ==================================================
# SETTINGS
# ==================================================

SAMPLE_RATE = 16000

TARGET_FRAMES = 600

MFCC_COEFFICIENTS = 13

# ==================================================
# AUDIO EXTRACTION
# ==================================================

def extract_audio_to_wav(
    video_path,
    wav_path
):

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        str(wav_path)
    ]

    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

# ==================================================
# MFCC
# ==================================================

def build_mfcc(
    wav_path
):

    audio, sr = sf.read(
        str(wav_path)
    )

    if audio.ndim > 1:
        audio = audio.mean(
            axis=1
        )

    audio = audio.astype(
        np.float32
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=MFCC_COEFFICIENTS
    )

    delta = librosa.feature.delta(
        mfcc
    )

    delta2 = librosa.feature.delta(
        mfcc,
        order=2
    )

    features = np.concatenate(
        [
            mfcc,
            delta,
            delta2
        ],
        axis=0
    )

    features = features.T

    return features

# ==================================================
# RESIZE
# ==================================================

def resize_mfcc(
    features,
    target_frames=TARGET_FRAMES
):

    num_frames = features.shape[0]

    if num_frames > target_frames:

        indices = np.linspace(
            0,
            num_frames - 1,
            target_frames
        ).astype(np.int32)

        features = features[
            indices
        ]

    elif num_frames < target_frames:

        padding = np.zeros(
            (
                target_frames - num_frames,
                features.shape[1]
            ),
            dtype=features.dtype
        )

        features = np.vstack(
            [
                features,
                padding
            ]
        )

    return features

# ==================================================
# MAIN API
# ==================================================
def preprocess_audio(video_path):

    fd, wav_path = tempfile.mkstemp(
        suffix=".wav"
    )

    os.close(fd)

    try:

        extract_audio_to_wav(
            video_path,
            wav_path
        )

        mfcc = build_mfcc(
            wav_path
        )

        mfcc = resize_mfcc(
            mfcc
        )

    finally:

        if os.path.exists(wav_path):
            os.remove(wav_path)

    mfcc = np.expand_dims(
        mfcc,
        axis=0
    )

    tensor = torch.tensor(
        mfcc,
        dtype=torch.float32
    )

    tensor = tensor.unsqueeze(0)

    return tensor
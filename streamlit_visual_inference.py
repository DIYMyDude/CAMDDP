#========================= streamlit_visual_inference.py =========================
#This file contains the supporting audio preprocessing of uploaded samples for the Streamlit inference/test web application of CAMDDP application.
#=================================================================================

import cv2
import numpy as np
import torch

# ==================================================
# SETTINGS
# ==================================================

NUM_FRAMES = 15

IMG_SIZE = (224, 224)

# ==================================================
# FRAME SAMPLING
# ==================================================

def get_target_frame_indices(
    total_frames,
    output_frames=15,
    chunk_size=10,
    frames_per_chunk=4
):

    if total_frames <= 0:
        return []

    indices = []

    chunk_count = max(
        1,
        (
            output_frames +
            frames_per_chunk - 1
        )
        // frames_per_chunk
    )

    if chunk_count == 1:

        starts = [0]

    else:

        max_start = max(
            0,
            total_frames - chunk_size
        )

        starts = []

        for i in range(chunk_count):

            pos = int(
                round(
                    i *
                    max_start /
                    (chunk_count - 1)
                )
            )

            starts.append(pos)

    for start in starts:

        chunk_frames = []

        for j in range(frames_per_chunk):

            idx = start + int(
                round(
                    j *
                    (chunk_size - 1)
                    /
                    (frames_per_chunk - 1)
                )
            )

            idx = min(
                max(idx, 0),
                total_frames - 1
            )

            chunk_frames.append(idx)

        indices.extend(chunk_frames)

    indices = indices[:output_frames]

    while len(indices) < output_frames:

        indices.append(
            indices[-1]
        )

    return indices

# ==================================================
# PREPROCESS
# ==================================================

def preprocess_visual(
    video_path,
    use_frame_difference=False
):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise ValueError(
            f"Unable to open video: {video_path}"
        )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if total_frames <= 0:

        cap.release()

        raise ValueError(
            f"Invalid frame count: {video_path}"
        )

    candidate_indices = get_target_frame_indices(
        total_frames=total_frames,
        output_frames=NUM_FRAMES * 3,
        chunk_size=10,
        frames_per_chunk=4
    )

    used = set(candidate_indices)

    for i in range(total_frames):

        if i not in used:

            candidate_indices.append(i)

    frames = []

    for frame_idx in candidate_indices:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_idx
        )

        ret, frame = cap.read()

        if not ret:
            continue

        frame = cv2.resize(
            frame,
            IMG_SIZE
        )

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        frame = (
            frame.astype(
                np.float32
            )
            / 255.0
        )

        frame = np.transpose(
            frame,
            (2, 0, 1)
        )

        frames.append(frame)

        if len(frames) >= NUM_FRAMES:
            break

    cap.release()

    if len(frames) == 0:

        raise ValueError(
            f"No frames extracted from video: {video_path}"
        )

    while len(frames) < NUM_FRAMES:

        frames.append(
            frames[-1]
        )

    frames = np.stack(
        frames,
        axis=0
    )

    if use_frame_difference:

        diff_frames = []

        for i in range(
            1,
            len(frames)
        ):
            diff = (
                frames[i]
                - frames[i - 1]
            )

            diff_frames.append(
                diff
            )


        if len(diff_frames) == 0:

            raise ValueError(
                "No frame differences generated."
            )


        frames = np.stack(
            diff_frames,
            axis=0
        )

    tensor = torch.tensor(
        frames,
        dtype=torch.float32
    )

    tensor = tensor.unsqueeze(0)

    return tensor
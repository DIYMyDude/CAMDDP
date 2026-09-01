#========================= streamlit_video_preview.py =========================
#This file contains the functions that converts originla video file into format decodable by web browser for preview in the Streamlit main web application of CAMDDP.
#==============================================================================

import subprocess
import tempfile


def needs_browser_conversion(
    video_path
):

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    codec = (
        result.stdout
        .strip()
        .lower()
    )

    return codec != "h264"


def create_preview_video(
    input_video
):

    preview_file = tempfile.NamedTemporaryFile(
        suffix=".mp4",
        delete=False
    )

    preview_file.close()

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_video),
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-c:a",
        "aac",
        preview_file.name
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    print(result.stderr)
    
    result.check_returncode()

    return preview_file.name

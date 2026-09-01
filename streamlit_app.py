#========================= streamlit_app.py =========================
#This file contains the Streamlit main web application interface for CAMDDP inference/test application.
#====================================================================

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from streamlit_predict import (
    load_model,
    predict_video,
    MODEL_CONFIGS
)

from streamlit_video_preview import (
    needs_browser_conversion,
    create_preview_video
)

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="CAMDDP",
    page_icon="🎭",
    layout="wide"
)

# ==================================================
# MODEL CACHE
# ==================================================

@st.cache_resource
def get_model(model_name):

    return load_model(model_name)

# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.title("CAMDDP")

    st.markdown(
        """
        Confidence-Aware Multimodal
        Deepfake Detection Pipeline
        """
    )

    st.divider()

    st.markdown(
        """
        **4-Class Deepfake Detection**

        A = Real Audio + Real Visual

        B = Fake Audio + Real Visual

        C = Real Audio + Fake Visual

        D = Fake Audio + Fake Visual
        """
    )

# ==================================================
# MAIN PAGE
# ==================================================

st.title(
    "CAMDDP Deepfake Detection"
)

# ==================================================
# MODEL SELECTION
# ==================================================

model_name = st.selectbox(
    "Select Trained Model",
    list(MODEL_CONFIGS.keys())
)

# ==================================================
# FILE UPLOAD
# ==================================================

uploaded_files = st.file_uploader(
    "Upload MP4 File(s)",
    type=["mp4"],
    accept_multiple_files=True
)

# ==================================================
# RUN BUTTON
# ==================================================

run_inference = st.button(
    "Run Inference",
    disabled=len(uploaded_files) == 0
)

# ==================================================
# INFERENCE
# ==================================================

results = []

if run_inference:

    with st.spinner(
        "Running CAMDDP inference..."
    ):

        model = get_model(
            model_name
        )

        use_frame_difference = (
            MODEL_CONFIGS[
                model_name
            ]["vdiff"]
        )

        progress_bar = st.progress(
            0
        )

        total_files = len(
            uploaded_files
        )

        for idx, uploaded_file in enumerate(
            uploaded_files
        ):

            suffix = (
                Path(
                    uploaded_file.name
                ).suffix
            )

            with tempfile.NamedTemporaryFile(
                suffix=suffix,
                delete=False
            ) as tmp:

                tmp.write(
                    uploaded_file.getbuffer()
                )

                temp_video_path = tmp.name

                preview_video_path = temp_video_path

                try:
                
                    if needs_browser_conversion(
                        temp_video_path
                    ):
                
                        preview_video_path = (
                            create_preview_video(
                                temp_video_path
                            )
                        )
                
                except Exception:
                
                    preview_video_path = temp_video_path

            try:

                result = predict_video(
                    temp_video_path,
                    model,
                    use_frame_difference
                )

                row = {

                    "Filename":
                        uploaded_file.name,

                    "_video_bytes":
                        Path(
                            preview_video_path
                        ).read_bytes(),

                    "Prediction":
                        result["prediction"],

                    "Description":
                        result["description"],

                    "Audio_Confidence":
                        round(
                            result["pa"],
                            4
                        ),

                    "Visual_Confidence":
                        round(
                            result["pv"],
                            4
                        ),

                    "Prob_A":
                        round(
                            result["prob_a"],
                            4
                        ),

                    "Prob_B":
                        round(
                            result["prob_b"],
                            4
                        ),

                    "Prob_C":
                        round(
                            result["prob_c"],
                            4
                        ),

                    "Prob_D":
                        round(
                            result["prob_d"],
                            4
                        ),

                    "Inference_Time":
                        round(
                            result["inference_time"],
                            3
                        )
                }

                results.append(
                    row
                )

            except Exception as e:

                results.append({

                    "Filename":
                        uploaded_file.name,

                    "_video_bytes":
                        uploaded_file.getvalue(),

                    "Prediction":
                        "ERROR",

                    "Description":
                        str(e),

                    "Audio_Confidence": None,
                    "Visual_Confidence": None,

                    "Prob_A": None,
                    "Prob_B": None,
                    "Prob_C": None,
                    "Prob_D": None,

                    "Inference_Time":
                        None
                })

            finally:
                Path(
                    temp_video_path
                ).unlink(
                    missing_ok=True
                )

                if (
                    preview_video_path
                    != temp_video_path
                ):

                    Path(
                        preview_video_path
                    ).unlink(
                        missing_ok=True
                    )


            progress_bar.progress(
                (idx + 1) / total_files
            )

# ==================================================
# RESULTS
# ==================================================

st.divider()

st.subheader(
    "Inference Results"
)

if len(results) > 0:

    results_df = pd.DataFrame(
        results
    )

    display_df = results_df.drop(
        columns=["_video_bytes"]
    )

    # ==========================================
    # SUMMARY
    # ==========================================

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Files Processed",
            len(results_df)
        )

    with col2:

        st.metric(
            "Model",
            model_name
        )

    with col3:

        st.metric(
            "Total Runtime (s)",
            round(
                results_df[
                    "Inference_Time"
                ]
                .fillna(0)
                .sum(),
                2
            )
        )

    # ==========================================
    # TABLE
    # ==========================================

    st.dataframe(
        display_df,
        use_container_width=True
    )

    # ==========================================
    # DOWNLOAD CSV
    # ==========================================

    csv_data = display_df.to_csv(
        index=False
    )

    st.download_button(
        label="Download Results CSV",
        data=csv_data,
        file_name="camddp_results.csv",
        mime="text/csv"
    )

    st.divider()

    st.subheader(
        "Video Preview"
    )

    for row in results:

        with st.expander(
            f"Preview: {row['Filename']}"
        ):

            col1, col2 = st.columns([1, 2])

            with col1:

                st.video(
                    row["_video_bytes"],
                    width=250
                )

            with col2:

                st.success(
                    f"Prediction: {row['Prediction']}"
                )

                st.write(
                    row["Description"]
                )

                st.write(
                    f"Audio Confidence (Pa): {row['Audio_Confidence']}"
                )

                st.write(
                    f"Visual Confidence (Pv): {row['Visual_Confidence']}"
                )


                st.write(
                    f"A={row['Prob_A']}, "
                    f"B={row['Prob_B']}, "
                    f"C={row['Prob_C']}, "
                    f"D={row['Prob_D']}"
                )

else:

    empty_df = pd.DataFrame(
        columns=[
            "Filename",
            "Prediction",
            "Description",
            "Audio_Confidence",
            "Visual_Confidence",
            "Prob_A",
            "Prob_B",
            "Prob_C",
            "Prob_D",
            "Inference_Time"
        ]
    )

    st.dataframe(
        empty_df,
        use_container_width=True
    )

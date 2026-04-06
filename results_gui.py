import os
import csv
from pathlib import Path

import streamlit as st
#streamlit run d:/NN/project/API/results_gui.py

st.set_page_config(page_title="Results Viewer", layout="wide")

st.title("📊 Transcription Results Viewer")
st.markdown(
    """
    **Note on Transcriptions:** 
    If the `ground_truth` looks like English gibberish (e.g., `waraj~aHa Alt~aqoriyru`), it is actually **Buckwalter Transliteration**, which is a standard way to represent Arabic letters using Latin ASCII characters! Deepgram is outputting the correct **Unicode Arabic** (`وَرَجَّحَ التَّقْرِيرُ`). They both say the exact same thing!
    """
)

# Paths
RESULTS_CSV = Path(__file__).parent / "results.csv"
TEST_GUI_DIR = Path(__file__).parent / "Test_for _gui"

if not RESULTS_CSV.exists():
    st.error(f"Could not find `{RESULTS_CSV}`. Please make sure the data script ran successfully.")
    st.stop()

# 1. Load the CSV results
results = []
try:
    with open(RESULTS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)
except Exception as e:
    st.error(f"Error reading CSV: {e}")
    st.stop()

if not results:
    st.warning("The results CSV is empty.")
    st.stop()

st.subheader("Results Data")

# Create a mapping of filename to row data
row_dict = {row["filename"]: row for row in results}
filenames = list(row_dict.keys())

# Let user pick a file
selected_filename = st.selectbox("Select an audio file from the results to visualize:", filenames)

if selected_filename:
    row_data = row_dict[selected_filename]
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Audio File")
        
        # Audio file path
        audio_path = TEST_GUI_DIR / selected_filename
        
        # Check if the wav file exists in Test_for_gui
        if audio_path.exists():
            audio_bytes = audio_path.read_bytes()
            st.audio(audio_bytes, format="audio/wav")
        else:
            # Fallback to the dataset if it's there
            fallback_dir = Path(__file__).parent.parent / "arabic-speech-corpus" / "test set" / "wav"
            fallback_path = fallback_dir / selected_filename
            if fallback_path.exists():
                st.info(f"Loaded audio from dataset (not found in Test_for _gui).")
                audio_bytes = fallback_path.read_bytes()
                st.audio(audio_bytes, format="audio/wav")
            else:
                st.warning(f"Audio file '{selected_filename}' could not be logically found.")

    with col2:
        st.subheader("Transcription Comparison")
        
        st.markdown("**Ground Truth (Buckwalter Transliteration):**")
        st.info(row_data.get("ground_truth", ""))
        
        st.markdown("**Deepgram Output (Arabic Unicode):**")
        deepgram_output = row_data.get("deepgram_transcript", "")
        
        if "ERROR" in deepgram_output.upper():
            st.error(deepgram_output)
        else:
            st.success(deepgram_output)
            
# Optional: Show a quick table overview 
with st.expander("Show Raw CSV Data"):
    st.dataframe(results)

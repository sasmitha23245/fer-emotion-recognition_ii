import streamlit as st
import subprocess
import tempfile
import os

st.title("Facial Emotion Recognition")

st.write(
"This application uses a pre-trained CNN model to detect emotions from facial expressions."
)

option = st.radio(
"Select Mode",
("Upload Image", "Open Webcam")
)

if option == "Upload Image":

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg","jpeg","png","bmp","tiff"]
    )

    if uploaded_file is not None:

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        temp_file.write(uploaded_file.read())
        temp_file.close()

        st.image(temp_file.name)

        if st.button("Detect Emotion"):
            subprocess.run(
                [r"c:\Users\Sasmika\Desktop\fer_project - Copy\venv\Scripts\python.exe", "detect_realtime.py", "--image", temp_file.name]
            )

elif option == "Open Webcam":

    if st.button("Start Webcam Detection"):
        subprocess.run([r"c:\Users\Sasmika\Desktop\fer_project - Copy\venv\Scripts\python.exe", "detect_realtime.py"])
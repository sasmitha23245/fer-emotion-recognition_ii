import streamlit as st
import subprocess
import tempfile
import os
import cv2
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Facial Emotion Recognition",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Exo+2:wght@400;600;700&display=swap');
    
    .main-header {
        text-align: center;
        padding: 50px 30px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        color: white;
        margin-bottom: 40px;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
        animation: slideDown 0.6s ease-out;
    }
    
    @keyframes slideDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .main-header h1 {
        font-family: 'Orbitron', sans-serif;
        font-size: 4.5em;
        font-weight: 900;
        margin: 0 auto;
        text-shadow: 3px 3px 12px rgba(0,0,0,0.4), 0 0 20px rgba(255,255,255,0.3);
        letter-spacing: 2px;
        line-height: 1.1;
        background: linear-gradient(135deg, #ffffff 0%, #e8d5ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-transform: uppercase;
        text-align: center;
        width: 100%;
        display: block;
    }
    
    .main-header p {
        font-family: 'Exo 2', sans-serif;
        font-size: 1.8em;
        font-weight: 600;
        margin: 20px 0 0 0;
        opacity: 0.95;
        letter-spacing: 0.5px;
        color: #ffe8ff;
    }
    
    .emotion-result {
        padding: 20px;
        border-radius: 10px;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        text-align: center;
        font-size: 1.8em;
        font-weight: bold;
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .section-title {
        font-size: 2em;
        font-weight: bold;
        margin-top: 30px;
        margin-bottom: 15px;
        color: #333;
        border-bottom: 3px solid #667eea;
        padding-bottom: 10px;
    }
    .upload-box {
        padding: 20px;
        border: 2px dashed #667eea;
        border-radius: 10px;
        background: #f8f9ff;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="main-header">
        <h1>Facial Emotion Recognition</h1>
        <p>🤖 AI-Powered Emotion Detection from Facial Expressions</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("### 🎛️ Settings")
mode = st.sidebar.radio(
    "Select Mode:",
    ("Upload Image", "🎥 Open Webcam"),
    help="Choose how you want to detect emotions"
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### About This App
This application uses a **Convolutional Neural Network (CNN)** 
to detect emotions from facial expressions with high accuracy.

**Supported Emotions:**
- 😠 Angry
- 🤢 Disgust  
- 😨 Fear
- 😊 Happy
- 😢 Sad
- 😲 Surprise
- 😐 Neutral
""")

# Main content
if mode == "Upload Image":
    st.markdown('<div class="section-title">📷 Upload & Analyze</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="upload-box">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["jpg","jpeg","png","bmp","tiff"],
            help="Select a clear photo with visible facial features"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Save temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        temp_file.write(uploaded_file.read())
        temp_file.close()
        
        with col1:
            st.markdown("### 📸 Original Image")
            st.image(temp_file.name, use_column_width=True, caption="Uploaded Image")
        
        with col2:
            st.markdown("### Detection Controls")
            
            detect_button = st.button(
                "Detect Emotion",
                type="primary",
                use_container_width=True,
                help="Click to analyze the facial emotion in the image"
            )
            
            if detect_button:
                # Processing indicator
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.info("⏳ Loading model... Please wait")
                progress_bar.progress(25)
                
                status_text.info("🔍 Analyzing facial features...")
                progress_bar.progress(50)
                
                status_text.info("🤖 Running emotion detection...")
                progress_bar.progress(75)
                
                # Run detection
                result = subprocess.run(
                    [r"c:\Users\Sasmika\Desktop\fer_project - Copy\venv\Scripts\python.exe", "detect_realtime.py", "--image", temp_file.name],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                progress_bar.progress(100)
                
                # Parse result from stdout
                output = result.stdout
                emotion_detected = None
                confidence = None
                
                for line in output.split('\n'):
                    if '[RESULT]' in line and 'Detected:' in line:
                        # Extract emotion and confidence
                        parts = line.split('Detected:')[1].strip().split('(')
                        emotion_detected = parts[0].strip()
                        if len(parts) > 1:
                            confidence = parts[1].replace('%)', '').strip()
                
                if emotion_detected:
                    status_text.empty()
                    progress_bar.empty()
                    
                    # Display result
                    st.markdown('<div class="emotion-result">', unsafe_allow_html=True)
                    st.markdown(f"**{emotion_detected}** ({confidence})")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.success("Detection Complete!")
                else:
                    st.warning("No faces detected in the image. Please try another image.")

elif mode == "🎥 Open Webcam":
    st.markdown('<div class="section-title">🎥 Real-Time Detection</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 📹 Webcam Instructions
        Click the button to open your webcam for real-time emotion detection:
        
        **Controls:**
        - **Q** - Quit/Close webcam
        - **S** - Save current frame
        
        **Note:** A separate window will open with the webcam feed and emotion detection overlay.
        """)
    
    with col2:
        st.markdown("### 🎮 Controls")
        if st.button(
            "▶️ Start Webcam Detection",
            type="primary",
            use_container_width=True,
            help="Opens webcam in a new window for real-time emotion detection"
        ):
            st.info("🎥 Webcam window is opening... Check your screen for the webcam feed!")
            
            with st.spinner("⏳ Running real-time detection..."):
                subprocess.run([r"c:\Users\Sasmika\Desktop\fer_project - Copy\venv\Scripts\python.exe", "detect_realtime.py"])
            
            st.success("✅ Webcam detection completed! Any saved frames are in your project folder.")
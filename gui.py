import tkinter as tk
from tkinter import filedialog
import subprocess
import os

# Path to the Python executable in the virtual environment
python_exe = r"c:/Users/Sasmika/Desktop/fer_project - Copy/venv/Scripts/python.exe"

def upload_image():
    # Open file dialog to select an image
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
    )
    if file_path:
        # Run the detection script with the selected image
        subprocess.run([python_exe, "detect_realtime.py", "--image", file_path])

def open_webcam():
    # Run the detection script in webcam mode
    subprocess.run([python_exe, "detect_realtime.py"])

# Create the main window
root = tk.Tk()
root.title("Facial Emotion Recognition")
root.geometry("520x320")  # Set a reasonable starting window size
root.minsize(600, 400)

# Make layout responsive
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=0)
root.rowconfigure(1, weight=0)
root.rowconfigure(2, weight=1)

# Title label
title_label = tk.Label(
    root,
    text="Emotion recognition from facial expressions",
    font=("Arial", 18, "bold"),
    anchor="center"
)
title_label.grid(row=0, column=0, pady=(30, 10), sticky="n")

# Description label
description_text = (
    "This application uses a pre-trained CNN model to detect emotions from facial expressions. "
    "You can upload a static image for analysis or open your webcam for real-time emotion detection."
)
desc_label = tk.Label(
    root,
    text=description_text,
    font=("Arial", 12),
    wraplength=480,
    justify="center",
    anchor="center"
)
desc_label.grid(row=1, column=0, padx=20, sticky="ew")

# Frame for buttons to center them
button_frame = tk.Frame(root)
button_frame.grid(row=2, column=0, pady=(25, 30), sticky="n")
button_frame.columnconfigure(0, weight=1)
button_frame.columnconfigure(1, weight=1)

# Upload Image button
upload_btn = tk.Button(
    button_frame,
    text="Upload Image",
    command=upload_image,
    font=("Arial", 12),
    width=16,
    height=2
)
upload_btn.grid(row=3, column=2, padx=(0, 15), sticky="e")

# Open Webcam button
webcam_btn = tk.Button(
    button_frame,
    text="Open Webcam",
    command=open_webcam,
    font=("Arial", 12),
    width=16,
    height=2
)
webcam_btn.grid(row=3, column=3, padx=(15, 0), sticky="w")

# Start the GUI event loop
root.mainloop()
import streamlit as st
import cv2
import tempfile
import os
from ultralytics import solutions

# Set up Streamlit app
st.title("People Counting for Drone Footage")
st.write("Upload a video to process it with object counting, then download the output.")

# Upload video
uploaded_file = st.file_uploader("Upload your video", type=["mp4", "avi", "mov"])

if uploaded_file:
    # Save uploaded video temporarily
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_input.write(uploaded_file.read())
    temp_input.close()

    # Read uploaded video
    cap = cv2.VideoCapture(temp_input.name)
    assert cap.isOpened(), "Error reading video file"
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    # Define region points
    region_points = [(0, 0), (w, 0), (w, h), (0, h)]  # region points for the whole image

    # Create a temporary output file
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    video_writer = cv2.VideoWriter(temp_output.name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # Initialize ObjectCounter
    counter = solutions.ObjectCounter(
        show=False,  # Disable GUI display for Streamlit
        region=region_points,
        model="visdrone_yolov11_model1.pt",
        classes=[0, 1]  # Specify classes to count (e.g., person and car)
    )

    # Process video
    progress_bar = st.progress(0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            st.write("Video processing completed.")
            break
        im0 = counter.count(im0)  # Perform object counting
        video_writer.write(im0)

        frame_count += 1
        progress_bar.progress(frame_count / total_frames)

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

    # Provide download link for processed video
    st.success("Processing complete! Download your processed video below.")
    with open(temp_output.name, "rb") as processed_file:
        st.download_button(
            label="Download Processed Video",
            data=processed_file,
            file_name="processed_video.mp4",
            mime="video/mp4"
        )

    # Clean up temporary files after download
    os.unlink(temp_input.name)
    os.unlink(temp_output.name)

import streamlit as st
import cv2
import tempfile
import os
from ultralytics import solutions

# Cache the model loading
@st.cache_data
def load_model():
    return solutions.ObjectCounter(
        show=False,  # Disable GUI display for Streamlit
        region=None,  # Will define dynamically
        model="visdrone_yolov11_model1.pt",
        classes=[0, 1]  # Specify classes to count (e.g., person and car)
    )

# Process video with a progress bar and cache the result
def process_video_with_progress(input_path, region_points, w, h, fps, total_frames):
    try:
        # Create a temporary output file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_output:
            temp_output_path = temp_output.name

        video_writer = cv2.VideoWriter(temp_output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # Initialize ObjectCounter
        counter = load_model()
        counter.region = region_points  # Set region dynamically

        # Read and process video with progress bar
        cap = cv2.VideoCapture(input_path)
        progress_bar = st.progress(0)  # Initialize the progress bar
        frame_count = 0

        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                break
            im0 = counter.count(im0)  # Perform object counting
            video_writer.write(im0)

            frame_count += 1
            progress_bar.progress(frame_count / total_frames)  # Update progress bar

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()
        progress_bar.empty()  # Clear the progress bar after completion
        return temp_output_path
    except Exception as e:
        st.error(f"An error occurred during video processing: {e}")
        return None

# Streamlit app logic
st.title("People Counting for Drone Footage")
st.write("Upload a video to process it with object counting, then download the output.")

# Initialize session state
if "processed_video_path" not in st.session_state:
    st.session_state.processed_video_path = None

# Upload video
uploaded_file = st.file_uploader("Upload your video", type=["mp4", "avi", "mov"])

if uploaded_file:
    # Save uploaded video temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input:
        temp_input.write(uploaded_file.read())
        temp_input_path = temp_input.name

    # Check if the video is already processed
    if st.session_state.processed_video_path is None or st.session_state.last_uploaded_file != uploaded_file.name:
        # Read uploaded video metadata
        cap = cv2.VideoCapture(temp_input_path)
        if not cap.isOpened():
            st.error("Error reading video file. Please try again with a valid video.")
            os.unlink(temp_input_path)
            st.stop()

        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Define region points dynamically
        region_points = [(0, 0), (w, 0), (w, h), (0, h)]

        # Process the video with a progress bar and store the path in session state
        st.info("Processing video. Please wait...")
        st.session_state.processed_video_path = process_video_with_progress(temp_input_path, region_points, w, h, fps, total_frames)
        st.session_state.last_uploaded_file = uploaded_file.name

    # Provide download link for processed video
    if st.session_state.processed_video_path:
        st.success("Processing complete! Download your processed video below.")
        with open(st.session_state.processed_video_path, "rb") as processed_file:
            st.download_button(
                label="Download Processed Video",
                data=processed_file,
                file_name="processed_video.mp4",
                mime="video/mp4"
            )
else:
    st.session_state.processed_video_path = None

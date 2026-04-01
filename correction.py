import cv2
import numpy as np
import os

# Load the saved calibration parameters
data_file = 'calibration_data.npz'
try:
    calib_data = np.load(data_file)
    mtx = calib_data['mtx']
    dist = calib_data['dist']
    print(f"Successfully loaded calibration data from '{data_file}'.")
except FileNotFoundError:
    print(f"Error: Could not find '{data_file}'. Please run camera_calibration.py first.")
    exit()

# Set video paths
input_video_path = 'chessboard.mp4'
output_video_path = 'distortion_correction_demo.mp4'

cap = cv2.VideoCapture(input_video_path)

if not cap.isOpened():
    print("Error: Could not open the input video.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Calculate 3 target frame indices to save as images (25%, 50%, and 75% of the video)
target_frames = [
    int(total_frames * 0.25),
    int(total_frames * 0.50),
    int(total_frames * 0.75)
]
saved_image_count = 0

# Output video width is doubled to concatenate two frames horizontally
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width * 2, frame_height))

print("Processing video and generating side-by-side comparison...")

# Calculate optimal camera matrix to preserve pixels
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (frame_width, frame_height), 1, (frame_width, frame_height))

current_frame = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Undistort the frame
    undistorted_frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)
    
    # Concatenate frames horizontally
    combined_frame = cv2.hconcat([frame, undistorted_frame])
    
    # Add text labels
    cv2.putText(combined_frame, 'Original', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.putText(combined_frame, 'Undistorted', (frame_width + 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
    
    # Save the combined frame as an image if the current frame matches a target frame
    if current_frame in target_frames:
        saved_image_count += 1
        image_filename = f'comparison_snapshot_{saved_image_count}.jpg'
        cv2.imwrite(image_filename, combined_frame)
        print(f"Saved snapshot image: {image_filename} (Frame {current_frame}/{total_frames})")
    
    out.write(combined_frame)
    current_frame += 1

cap.release()
out.release()
print(f"Video processing complete. Output saved to '{output_video_path}'.")
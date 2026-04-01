import cv2
import numpy as np

# Test various dimensions. (9, 6) is the mathematical target, but we check variations.
TEST_DIMENSIONS = [(9, 6), (6, 9), (8, 6), (10, 7)]

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

calibration_video_path = 'chessboard.mp4'
cap = cv2.VideoCapture(calibration_video_path)

if not cap.isOpened():
    print("Error: Could not open the calibration video.")
    exit()

print("Extracting frames and performing brute-force corner search...")

frame_count = 0
sample_interval = 15
valid_frame_shape = None

objpoints = []
imgpoints = []

# Removed CALIB_CB_FAST_CHECK because it fails easily with text on the paper or high resolution
find_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

detected_board_size = None
objp = None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    if frame_count % sample_interval == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if valid_frame_shape is None:
            valid_frame_shape = gray.shape[::-1]
        
        # If we haven't locked onto a board size yet, try all variations
        sizes_to_try = [detected_board_size] if detected_board_size else TEST_DIMENSIONS
        
        for board_size in sizes_to_try:
            ret_corners, corners = cv2.findChessboardCorners(gray, board_size, find_flags)
            
            if ret_corners:
                # Lock onto this size if it's the first time finding it
                if not detected_board_size:
                    print(f"Bingo! Locked onto board size: {board_size}")
                    detected_board_size = board_size
                    
                    # Initialize object points dynamically based on the locked size
                    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
                    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
                    
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
                
                # Draw the corners
                cv2.drawChessboardCorners(frame, board_size, corners2, ret_corners)
                print(f"Found corners in frame {frame_count}")
                break # Stop trying other sizes for this frame
        
        # Resize display window so it fits on screen, even if the video is 4K
        display_frame = cv2.resize(frame, (800, 600))
        cv2.imshow('Calibration Processing', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

if not objpoints:
    print("Error: Could not find chessboard corners in the video.")
    print("Check if the video resolution is too high or lighting is too harsh.")
    exit()

print(f"\nSuccessfully collected points from {len(objpoints)} frames.")
print("Calculating camera parameters... This may take a moment.")

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, valid_frame_shape, None, None)

print("\n--- Calibration Results ---")
print(f"RMSE (Reprojection Error): {ret:.4f}")
print(f"fx: {mtx[0, 0]:.4f}")
print(f"fy: {mtx[1, 1]:.4f}")
print(f"cx: {mtx[0, 2]:.4f}")
print(f"cy: {mtx[1, 2]:.4f}")
print("Distortion Coefficients:", dist.ravel())
print("---------------------------\n")

output_data_file = 'calibration_data.npz'
np.savez(output_data_file, mtx=mtx, dist=dist, ret=ret)
print(f"Calibration parameters saved to '{output_data_file}' successfully.")
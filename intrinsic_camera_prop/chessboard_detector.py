import numpy as np
import cv2

# List of calibration image paths
calibration_image_paths = [
  #  r'C:\Users\Luke Hamilton\Pictures\Camera Roll\chessboard_laptop_cam.jpg',
    r'C:\Biorobotics\Dissertation\visual servo control\camera calibration\chessboard-calibration-USB\image_1.jpg',
    r'C:\Biorobotics\Dissertation\visual servo control\camera calibration\chessboard-calibration-USB\image_2.jpg',
    r'C:\Biorobotics\Dissertation\visual servo control\camera calibration\chessboard-calibration-USB\image_3.jpg',
    r'C:\Biorobotics\Dissertation\visual servo control\camera calibration\chessboard-calibration-USB\image_4.jpg',
    r'C:\Biorobotics\Dissertation\visual servo control\camera calibration\chessboard-calibration-USB\image_5.jpg',
    r'C:\Biorobotics\Dissertation\visual servo control\camera calibration\chessboard-calibration-USB\image_6.jpg',
    r'C:\Biorobotics\Dissertation\visual servo control\camera calibration\chessboard-calibration-USB\image_7.jpg',
    r'C:\Biorobotics\Dissertation\visual servo control\camera calibration\chessboard-calibration-USB\image_8.jpg',
    r'C:\Biorobotics\Dissertation\visual servo control\camera calibration\chessboard-calibration-USB\image_9.jpg',
    r'C:\Biorobotics\Dissertation\visual servo control\camera calibration\chessboard-calibration-USB\image_10.jpg',
    r'C:\Biorobotics\Dissertation\visual servo control\camera calibration\chessboard-calibration-USB\image_11.jpg',
    r'C:\Biorobotics\Dissertation\visual servo control\camera calibration\chessboard-calibration-USB\image_12.jpg',
    r'C:\Biorobotics\Dissertation\visual servo control\camera calibration\chessboard-calibration-USB\image_13.jpg',
    # Add more paths as needed
]

# Define the size of the calibration pattern
pattern_size = (7, 7)  # Number of inner corners in the calibration pattern

# Arrays to store object points and image points from all the calibration images
obj_points_list = []  # 3D points in real-world space
img_points_list = []  # 2D points in image plane

# Detect corners in the calibration images
for img_path in calibration_image_paths:
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the corners in the calibration pattern
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    if ret:
        obj_points = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        obj_points[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        obj_points_list.append(obj_points)
        img_points_list.append(corners)

        # Draw the corners on the image
        cv2.drawChessboardCorners(img, pattern_size, corners, ret)

        print("Chessboard detected in:", img_path)
    else:
        print("Chessboard not detected in:", img_path)

    # Display the image with corners
    cv2.imshow("Detected Corners", img)
    cv2.waitKey(0)

# Perform camera calibration
ret, camera_matrix, distortion_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    obj_points_list, img_points_list, gray.shape[::-1], None, None
)

# Save the camera matrix and distortion coefficients to a file
np.savez(r'C:\Biorobotics\Dissertation\visual servo control\camera calibration\camera_matrix.npz', camera_matrix=camera_matrix, distortion_coeffs=distortion_coeffs)



cv2.destroyAllWindows()

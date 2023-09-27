from dotenv import load_dotenv
# load the environment variables (in this case opencv frame read attempts)
load_dotenv()

import os
print(os.environ['OPENCV_FFMPEG_READ_ATTEMPTS'])
import cv2
from ultralytics import YOLO

# Load model
model = YOLO(r"E:\University\Bristol\Dissertation\Roboflow\YOLO\YOLOv8\best.pt", task="detect")
print('Model loaded')

# Set model parameters
model.conf = 0.75 # NMS confidence threshold
model.iou = 0.45 # NMS IoU threshold
model.classes = None # (optional list) filter by class
print('Model values set')

# Open the video file
video_path = r"E:\University\Bristol\Dissertation\Roboflow\self-sourced\video_source.mp4"
cap = cv2.VideoCapture(video_path)

# Define the output video parameters
output_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
output_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter('output_video.avi', fourcc, 30, (output_width, output_height))

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Write the annotated frame to the output video
        output_video.write(annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture and output video objects
cap.release()
output_video.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
    
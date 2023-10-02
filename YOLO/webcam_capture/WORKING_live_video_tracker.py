from dotenv import load_dotenv
load_dotenv()

import os
import cv2
from ultralytics import YOLO

# Load model
model = YOLO(r"E:\University\Bristol\Dissertation\Roboflow\YOLO\YOLOv8\best.pt", task="detect")
print('Model loaded')

# # Set model parameters
# model.conf = 0.85  # NMS confidence threshold
# model.iou = 0.45   # NMS IoU threshold
# model.classes = None  # (optional list) filter by class
# print('Model values set')

# Open the camera (0 is the default camera, change it if necessary)
cap = cv2.VideoCapture(0)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Loop through the live video frames
while True:
    # Read a frame from the camera
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, save_conf=True)
        print(results)

        # predictions = model.predictor(frame)
        # print(predictions)
        # class_probs = predictions.pred[0]['pred']
        # print(class_probs)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("Custom Model Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        print("Error: Could not read frame.")
        break

# Release the camera and close the display window
cap.release()
cv2.destroyAllWindows()

from dotenv import load_dotenv
load_dotenv()
import cv2
from ultralytics import YOLO

# Load model
model = YOLO(r"E:\University\Bristol\Dissertation\Roboflow\YOLO\YOLOv8\best.pt", task="detect")
print('Model loaded')

# Open the camera (0 is the default camera, change it if necessary)
cap = cv2.VideoCapture(0)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# # Get the frames per second (FPS) of the video streamq
# fps = int(cap.get(cv2.CAP_PROP_FPS))
# print(f"FPS: {fps}")

# Initialize a color variable for drawing bounding boxes (for the TARGET FLOWER) (e.g., blue)
bbox_color = (255, 0, 0)  # Blue color (BGR format)

# Loop through the live video frames
while True:
    # Read a frame from the camera
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)
        #print(results)

        # Visualize the results on the frame (must be done first to allow for target flower to be drawn on top)
        annotated_frame = results[0].plot()

        # Initialize variables to track the highest confidence and associated information
        highest_confidence = 0.0
        highest_confidence_bbox = None

        # Iterate through the tracked objects in the results list
        for r in results:
            # Check if there are detections in the frame
            if len(r.boxes) > 0: 
                #print(r.boxes)
                confidence_values = r.boxes.conf  # Get the confidence values for the tracked objects
                highest_confidence_index = confidence_values.argmax()  # Get the index of the highest confidence

                if confidence_values[highest_confidence_index] > highest_confidence:
                    highest_confidence = confidence_values[highest_confidence_index]
                    highest_confidence_bbox = r.boxes[highest_confidence_index]
                    #print(highest_confidence_bbox)
            else:
                print('No detections')
                break

        # Draw a bounding box around the object with the highest confidence (if found)
        if highest_confidence_bbox is not None:
            x_min, y_min, x_max, y_max = highest_confidence_bbox.xyxy[0]  # Get the bounding box coordinates
            cv2.rectangle(annotated_frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), bbox_color, 2)  # Draw a blue bounding box

        
        # Add a label to the bounding box
        label_text = "TARGET FLOWER"
        label_font = cv2.FONT_HERSHEY_SIMPLEX
        label_font_scale = 0.5
        label_font_color = (0, 0, 0)  # White color (BGR format)
        label_thickness = 2

        # Calculate the position for the label (above the bounding box)
        label_x = int(x_min) - 10
        label_y = int(y_min) - 10  # Adjust the vertical position as needed

        # Draw the label on the frame
        cv2.putText(annotated_frame, label_text, (label_x, label_y), label_font, label_font_scale, label_font_color, label_thickness)

        # # Display the frame
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

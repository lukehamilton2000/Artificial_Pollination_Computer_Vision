import cv2
from ultralytics import YOLO

def capture_image():
    camera = cv2.VideoCapture(0)
    # Allow camera sensor time to warm up
    ret, frame = camera.read()
    camera.release()
    if not ret:
        raise RuntimeError("Failed to capture image from the camera.")
    # Get frame size
    height, width, _ = frame.shape

    # Take left half of image
    #left_half_img = frame[:, :width//2, :]

    # Resize the image to fit within 640 x 640 pixels
    final_img = cv2.resize(frame, (640, 640))
    return final_img

# Load model
model = YOLO(r"E:\University\Bristol\Dissertation\Roboflow\YOLO\YOLOv8\best.pt", task="detect")
print('Model loaded')

# Set model parameters
model.conf = 0.75 # NMS confidence threshold
model.iou = 0.45 # NMS IoU threshold
model.classes = None # (optional list) filter by class
print('Model values set')


# Create a window to display the camera feed
cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)

while True:
    # Capture and save the image
    imgs = capture_image()

    # Inference with smaller input size and test time augmentation
    results_list = model(imgs, imgsz=640)

    # Access detection results from the Boxes object
    for results in results_list:
        detection_results = results.xyxy[0].cpu().numpy()  # Assuming the detection results are stored directly in xyxy attribute
        detection_results = detection_results[detection_results[:, 4] > model.conf]

        # Display the image with bounding boxes
        display_img = imgs.copy()
        for row in detection_results:
            x_min, y_min, x_max, y_max, conf, cls = row
            label = f"{model.names[int(cls)]} {conf:.2f}"
            cv2.rectangle(display_img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
            cv2.putText(display_img, label, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the image in the window
        cv2.imshow("Object Detection", display_img)

        # Check for key press to end the loop (wait for 10 ms)
        key = cv2.waitKey(10)
        if key == 27:  # Press 'Esc' to exit
            break

# Release resources
cv2.destroyAllWindows()
import pprint
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

while True:
    
    # Capture and save the image
    imgs = capture_image()

    # Inference with smaller input size and test time augmentation
    results = model(imgs, imgsz=640)

    numpy_results = results.orig_img
    numpy_results.print
    numpy_results.xyxy[0]


# Release resources
cv2.destroyAllWindows()




# # Parse results
# predictions = results.pred[0]
# boxes = predictions[:, :4] # x1, y1, x2, y2
# scores = predictions[:, 4]
# categories = predictions[:, 5]

# # Extract x1, y1, x2, and y2 to their own variables
# x1, y1, x2, y2 = boxes[0].cpu().numpy() if boxes.shape[0] == 1 else (None, None, None, None)

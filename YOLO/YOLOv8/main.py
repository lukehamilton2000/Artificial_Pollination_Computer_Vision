
from roboflow import Roboflow
rf = Roboflow(api_key="yoV7Fcn9ZtuEEvCfJ74L")
project = rf.workspace("dissertation-qmii2").project("flower-detection-online-source")
dataset = project.version(1).download("yolov8")


# create object model as Yolo Version 8
from ultralytics import YOLO

model = YOLO(r'E:\University\Bristol\Dissertation\Roboflow\YOLO\YOLOv8\yolov8n.pt')





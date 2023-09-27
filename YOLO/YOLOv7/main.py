from roboflow import Roboflow

rf = Roboflow(api_key="yoV7Fcn9ZtuEEvCfJ74L")
project = rf.workspace("dissertation-qmii2").project("flower-detection-cvybj")
dataset = project.version(5).download("yolov7")

from ultralytics import YOLO

model = YOLO(r"E:\University\Bristol\Dissertation\Roboflow\YOLO\YOLOv7\yolov7\cfg\deploy\yolov7.yaml")
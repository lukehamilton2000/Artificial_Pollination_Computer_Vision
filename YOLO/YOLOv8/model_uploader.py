
from roboflow import Roboflow
rf = Roboflow(api_key="yoV7Fcn9ZtuEEvCfJ74L")
project = rf.workspace("dissertation-qmii2").project("flower-detection-cvybj")
#dataset = project.version(15).download("yolov8")

version = project.version(15)
version.deploy("yolov8", r"E:\University\Bristol\Dissertation\Roboflow\YOLO\YOLOv8\runs\detect\epoch_100")



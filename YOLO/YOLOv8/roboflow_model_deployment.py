from roboflow import Roboflow

# create a roboflow object (download the dataset)
rf = Roboflow(api_key="yoV7Fcn9ZtuEEvCfJ74L")
project = rf.workspace("dissertation-qmii2").project("flower-detection-cvybj")

project.version(3).deploy(model_type="yolov8", model_path=r"E:\University\Bristol\Dissertation\Roboflow\YOLO\YOLOv8\runs\detect\train3")
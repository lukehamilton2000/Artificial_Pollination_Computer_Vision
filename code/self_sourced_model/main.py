from roboflow import Roboflow
rf = Roboflow(api_key="yoV7Fcn9ZtuEEvCfJ74L")
project = rf.workspace().project("flower-detection-cvybj")
model = project.version(2).model

# infer on a local image
print(model.predict(r"E:\University\Bristol\Dissertation\Roboflow\test.jpg", confidence=40, overlap=30).json())

# visualize your prediction
model.predict(r"E:\University\Bristol\Dissertation\Roboflow\test.jpg", confidence=40, overlap=30).save("prediction.jpg")

# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())
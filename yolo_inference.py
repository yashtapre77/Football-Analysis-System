from ultralytics import YOLO

model = YOLO('yolov8n')  # Load model

results = model.predict('inp_vid/train1.mp4', save=True)  # Inference

print("+++++++++++++++++++++++++")
print(results[0])
print("+++++++++++++++++++++++++")
for box in results[0].boxes:
    print(box)
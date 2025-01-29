from ultralytics import YOLO

model = YOLO('models/best.pt')  # Load model

# Specify the directory where the output video will be saved
results = model.predict('inp_vid/train1.mp4', save=True, save_dir='runs/detect/predict')  # Inference with custom save path


print("+++++++++++++++++++++++++")
print(results[0])
print("+++++++++++++++++++++++++")
for box in results[0].boxes:
    print(box)
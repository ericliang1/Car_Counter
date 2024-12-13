from ultralytics import YOLO

model = YOLO("runs/detect/train4/weights/best.pt")  

results = model.predict(source='test_images', conf=0.3, save=True)  
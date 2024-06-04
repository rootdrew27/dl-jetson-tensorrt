from ultralytics import YOLO

import torch
print(f'\nTorch Available : {torch.cuda.is_available()}\n')

model = YOLO('yolov8n.pt')

results = model.train(data='VOC.yaml', epochs=100, imgsz=640)

print(results)

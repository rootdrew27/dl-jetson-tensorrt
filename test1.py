from ultralytics import YOLO
from ultralytics.utils.benchmarks import benchmark
import torch

print(f'Torch is available: {torch.cuda.is_available()}\n')


# Load Pytorch model 
model = YOLO('yolov8n.pt')

model.export(format='engine') #creates (ie exports)  yolov8n.engine

# Load the exported model
trt_model = YOLO('yolov8n.engine')

# Run Inference
results = trt_model.val()

print(results)

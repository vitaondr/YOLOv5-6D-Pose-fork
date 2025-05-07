
import torch
ckpt = torch.load('YOLOv5-6D-Pose-fork/runs/train/exp19/weights/best.pt')
print(ckpt['best_fitness'])
print(ckpt['training_results'])
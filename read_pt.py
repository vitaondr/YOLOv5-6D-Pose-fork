
import torch

path = "runs/train/exp23/weights/best.pt"

checkpoint = torch.load(path)

print(checkpoint.keys())
print(checkpoint['epoch'])
print(checkpoint['best_fitness'])
print(checkpoint['training_results'])
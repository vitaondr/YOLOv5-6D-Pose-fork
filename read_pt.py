
import torch

path = "runs/train/exp4/weights/last.pt"

checkpoint = torch.load(path)

print(checkpoint.keys())
print(checkpoint['epoch'])
print(checkpoint['best_fitness'])
print(checkpoint['training_results'])
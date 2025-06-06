import wandb
import pandas as pd
import os



# run_path = "vitaondr-czech-technical-university-in-prague/YOLOv5/mwuozb1c" # real dataset
run_path = "vitaondr-czech-technical-university-in-prague/YOLOv5/gnb7f5ps" # synthetic dataset
csv_path = "metrics.csv"

train_metrics = ['train/obj_loss', 'train/box_loss', 'train/cls_loss',  # train loss
                        'x/lr0', 'x/lr1', 'x/lr2', 'train/mean_loss', 'epoch']
val_metrics = ['val/mean_corner_err_2d', 'val/acc', 'val/acc3d', 'val/acc10cm10deg', # val metrics
                         'val/obj_loss', 'val/box_loss', 'val/cls_loss','val/mean_loss', 'val/total_loss']

folders = ['train', 'val', 'x']
run_name = "exp3"

wandb.login()

api = wandb.Api()

# get data from wandb
run = api.run(run_path)

for folder in folders:
    # Create the directory if it doesn't exist
    os.makedirs(f"metrics/{run_name}/{folder}", exist_ok=True)
    print(f"Created directory: metrics/{run_name}/{folder}")

for metric in train_metrics + val_metrics:
    df = run.history(keys=[metric])
    # Ensure the directory exists
    df.to_csv(f"metrics/{run_name}/{metric}.csv", index=False)




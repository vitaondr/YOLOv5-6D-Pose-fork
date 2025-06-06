import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

csv_folder = "metrics"
run_name = "exp2"

train_metrics = ['train/obj_loss', 'train/box_loss', 'train/cls_loss',  # train loss
                        'x/lr0', 'x/lr1', 'x/lr2', 'train/mean_loss', 'epoch']
val_metrics = ['val/mean_corner_err_2d', 'val/acc', 'val/acc3d', 'val/acc10cm10deg', # val metrics
                         'val/obj_loss', 'val/box_loss', 'val/cls_loss','val/mean_loss', 'val/total_loss']



metric_dict = {}
for metric in train_metrics + val_metrics:
   
   df = pd.read_csv(f"{csv_folder}/{run_name}/{metric}.csv")
   metric_dict[metric] = df

# create plot for mean_loss
train_mean_loss_df = metric_dict['train/mean_loss']
val_mean_loss_df = metric_dict['val/mean_loss']
epoch_df = metric_dict['epoch']

df_merged_val = pd.merge_asof(val_mean_loss_df, epoch_df, on="_step", direction="backward")
df_merged_train = pd.merge_asof(train_mean_loss_df, epoch_df, on="_step", direction="nearest")

min_val_loss_epoch = df_merged_val.loc[df_merged_val['val/mean_loss'].idxmin(), 'epoch']
print(f"Epoch with minimal val/mean_loss: {min_val_loss_epoch}")
print(df_merged_train['train/mean_loss'][206])
print(df_merged_train['epoch'].unique())
print(df_merged_val['epoch'].unique())

print("plotting...")
plt.figure(figsize=(10, 5))
plt.plot(df_merged_train['epoch'], df_merged_train['train/mean_loss'], label='train/mean_loss') 
plt.plot(df_merged_val['epoch'], df_merged_val['val/mean_loss'], label='val/mean_loss')

# Mark the epoch with minimal val/mean_loss on the plot
plt.axvline(x=min_val_loss_epoch, color='r', linestyle='--', label=f'Min Val Loss Epoch ({min_val_loss_epoch})')

plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Mean Loss', fontsize=14)
# Increase the font size of the numbers on the axis
plt.tick_params(axis='both', which='major', labelsize=14)
plt.legend(fontsize=14)
plt.grid()

plt.show()




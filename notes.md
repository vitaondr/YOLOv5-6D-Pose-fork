# SSH connection to cluster

ssh vitaondr@login3.rci.cvut.cz

## start the interactive window 

srun -p gpufast --gres=gpu:1 --mem=30000 --pty bash -i

### Build singularity image

singularity build singularity.sif singularity.def # execute in the Yolo repo

## start singularity on the cluster 

singularity exec --nv singularity.sif bash

## train the model

python3.9 train.py --batch 32 --epochs 5 --cfg models/yolov5s_6dpose_bifpn.yaml --hyp configs/hyp.single.yaml --weights ../data/weights/yolov5s.pt --data configs/drone_data/drone.yaml --rect --cache --optimizer Adam


## this is how to strat a job 

sbatch -n1 -o train_job.stdout cls_train.sh
sbatch -n1 -o train_job.stdout cls_train_real.sh

## look in the queue

squeue


## watch if working 

watch -n 0.5 "squeue | grep vitao"

## watch progress 

tail -f runs/cls_train/command1.out 
tail -f runs/cls_train/command1_real.out 
tail -f runs/cls_train/command1_resume.out


## to resume training, resume_training has to be modified and then played

sbatch -n1 -o train_job.stdout resume_cls_train.sh

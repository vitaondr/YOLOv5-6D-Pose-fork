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



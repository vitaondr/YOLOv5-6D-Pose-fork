#!/usr/bin/env bash
python3.9 train.py --batch 32 --epochs 1000 --cfg models/yolov5s_6dpose_bifpn.yaml --hyp configs/hyp.single.yaml --weights ../data/weights/yolov5s.pt --data configs/drone_data/drone_real.yaml --rect --cache --optimizer Adam

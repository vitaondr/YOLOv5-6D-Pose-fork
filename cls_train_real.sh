#!/bin/sh

# options preceded with #SBATCH must be before any executable commands in the script

#SBATCH --job-name=yolo6D_train                                			# just to recognize the job in the queue overview
#SBATCH --nodes=1
#SBATCH --ntasks=3                     						# total number of tasks (total number of srun commands)
#SBATCH --gres=gpu:1		                                    		# generic consumable resources, see https://slurm.schedmd.com/sbatch.html
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=30GB
#SBATCH --partition=gpulong						                      	# partition name
#SBATCH --output=YOLOv5-6D-Pose-fork/runs/cls_train/yolo6D_train.out   		    # standard output file
#SBATCH --error=YOLOv5-6D-Pose-fork/runs/cls_train/yolo6D_train.out    		    # standard error file
#SBATCH --time=60:00:00							                        # time limit

PATH_TO_YOLO6D="$HOME/YOLOv5-6D-Pose-fork"
PATH_TO_YOLO6D_IMG="$HOME/YOLOv5-6D-Pose-fork/singularity.sif"

# COMMAND1="$PATH_TO_YOLO6D/train.sh"
COMMAND1="$PATH_TO_YOLO6D/train_real.sh"
COMMAND1_OUT="$PATH_TO_YOLO6D/runs/cls_train/command1.out"

echo "running batch job"
srun --ntasks=1 --gres=gpu:1 singularity run --nv $PATH_TO_YOLO6D_IMG $COMMAND1 > $COMMAND1_OUT 2>&1 &   # "&" symbol is used to run commands simultaneously
wait        # without wait command, 1. task would cancel itself, given 2. task completed successfully

echo "batch job done"

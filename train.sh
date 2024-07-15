export CUDA_VISIBLE_DEVICES=1,2,3,4,5,8,9
export WORLD_SIZE=6
export MASTER_ADDR='localhost'
export MASTER_PORT=25002
export LOCAL_RANK=0

torchrun --nproc_per_node=6 train.py
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WORLD_SIZE=8
export MASTER_ADDR='localhost'
export MASTER_PORT=25002
export LOCAL_RANK=0

torchrun --nproc_per_node=$WORLD_SIZE train.py
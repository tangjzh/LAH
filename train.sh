export CUDA_VISIBLE_DEVICES=4,5,6,7
export WORLD_SIZE=4
export MASTER_ADDR='localhost'
export MASTER_PORT=25002
export LOCAL_RANK=4

torchrun --nproc_per_node=$WORLD_SIZE train.py
export CUDA_VISIBLE_DEVICES=1,2,3,4
export WORLD_SIZE=4
export MASTER_ADDR='localhost'
export MASTER_PORT=25002
export LOCAL_RANK=0

torchrun --nproc_per_node=$WORLD_SIZE --master_port=$MASTER_PORT train.py --config configs/lah.yaml
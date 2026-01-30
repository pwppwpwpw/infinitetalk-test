export CUDA_VISIBLE_DEVICES='0,1,2,3'
GPU_NUM=4
#export CUDA_VISIBLE_DEVICES='0,1'
#GPU_NUM=2
torchrun --nproc_per_node=$GPU_NUM --standalone infinitetalk_parallel_pipeline.py --ulysses_size=$GPU_NUM

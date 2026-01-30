export CUDA_VISIBLE_DEVICES='7'
GPU_NUM=1
#export CUDA_VISIBLE_DEVICES='1,2,3,4'
#GPU_NUM=4
torchrun --nproc_per_node=$GPU_NUM --standalone generate_infinitetalk.py \
    --ckpt_dir weights/Wan2.1-I2V-14B-480P \
    --dit_path weights/aniWan2114BFp8E4m3fn_i2v480pNew.safetensors \
    --infinitetalk_dir weights/Wan2_1-InfiniTetalk-Single_fp16.safetensors \
    --wav2vec_dir 'weights/chinese-wav2vec2-base' \
    --lora_dir weights/Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors \
    --input_json examples/1.json \
    --lora_scale 1.0 \
    --ulysses_size=$GPU_NUM \
    --size infinitetalk-480 \
    --sample_text_guide_scale 1.0 \
    --sample_audio_guide_scale 1.0 \
    --offload_model false \
    --sample_steps 4 \
    --mode streaming \
    --motion_frame 9 \
    --sample_shift 11 \
    --color_correction_strength 0.0 \
    --save_file save_res/${GPU_NUM}gpu_lightx2v_4step \
    --frame_num 97 \
    --max_frame_num 9999999

    # --infinitetalk_dir weights/InfiniteTalk/single/infinitetalk.safetensors \
    # --infinitetalk_dir weights/InfiniteTalk/comfyui/infinitetalk_single.safetensors \
    # --infinitetalk_dir weights/Wan2_1-InfiniteTalk-Single_fp8_e5m2_scaled_KJ.safetensors \
    # --infinitetalk_dir weights/Wan2_1-InfiniTetalk-Single_fp16.safetensors \
    # --dit_path weights/Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors \
    # --dit_path weights/aniWan2114BFp8E4m3fn_i2v480pNew.safetensors \

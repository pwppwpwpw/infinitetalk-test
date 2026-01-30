# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import glob
import logging
import os
import sys
import json
import warnings
from datetime import datetime
import time
import random
import torch
import torch.distributed as dist
from types import SimpleNamespace
from pathlib import Path
import numpy as np
from einops import rearrange
import soundfile as sf
import subprocess
import librosa
import pyloudnorm as pyln
import shutil

# 假设这些包在环境中已安装
import wan
from wan.configs import SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.utils.utils import str2bool, is_video, split_wav_librosa
from wan.utils.multitalk_utils import save_video_ffmpeg, rand_name, cache_video
from kokoro import KPipeline
from transformers import Wav2Vec2FeatureExtractor
from src.audio_analysis.wav2vec2 import Wav2Vec2Model
from wan.utils.segvideo import shot_detect
from sr_pipe_trt import VideoSuperResolution
from audio_utils import simple_cut_audio, vocal_separate, save_audio_execute
warnings.filterwarnings('ignore')

# ---------------- Utility Functions (Stateless) ---------------- #
def loudness_norm(audio_array, sr=16000, lufs=-23):
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio_array)
    if abs(loudness) > 100:
        return audio_array
    normalized_audio = pyln.normalize.loudness(audio_array, loudness, lufs)
    return normalized_audio

def extract_audio_from_video(filename, sample_rate):
    raw_audio_path = filename.split('/')[-1].split('.')[0]+'.wav'
    # 注意：在服务中并发调用时，文件名冲突可能导致问题，建议改用 uuid 或 tempfile
    import uuid
    raw_audio_path = f"/tmp/{uuid.uuid4()}.wav"

    ffmpeg_command = [
        "ffmpeg", "-y", "-i", str(filename), "-vn",
        "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "2",
        str(raw_audio_path),
    ]
    subprocess.run(ffmpeg_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    human_speech_array, sr = librosa.load(raw_audio_path, sr=sample_rate)
    human_speech_array = loudness_norm(human_speech_array, sr)
    if os.path.exists(raw_audio_path):
        os.remove(raw_audio_path)
    return human_speech_array

def audio_prepare_single(audio_path, sample_rate=16000):
    ext = os.path.splitext(audio_path)[1].lower()
    if ext in ['.mp4', '.mov', '.avi', '.mkv']:
        human_speech_array = extract_audio_from_video(audio_path, sample_rate)
        return human_speech_array
    else:
        human_speech_array, sr = librosa.load(audio_path, sr=sample_rate)
        human_speech_array = loudness_norm(human_speech_array, sr)
        return human_speech_array


def audio_prepare_multi(left_path, right_path, audio_type, sample_rate=16000):
    # 此处逻辑保持原样，仅做少量防错处理
    if not (left_path == 'None' or right_path == 'None'):
        human_speech_array1 = audio_prepare_single(left_path)
        human_speech_array2 = audio_prepare_single(right_path)
    elif left_path == 'None':
        human_speech_array2 = audio_prepare_single(right_path)
        human_speech_array1 = np.zeros(human_speech_array2.shape[0])
    elif right_path == 'None':
        human_speech_array1 = audio_prepare_single(left_path)
        human_speech_array2 = np.zeros(human_speech_array1.shape[0])

    # Ensure lengths match
    max_len = max(len(human_speech_array1), len(human_speech_array2))
    if len(human_speech_array1) < max_len:
         human_speech_array1 = np.pad(human_speech_array1, (0, max_len - len(human_speech_array1)))
    if len(human_speech_array2) < max_len:
         human_speech_array2 = np.pad(human_speech_array2, (0, max_len - len(human_speech_array2)))

    if audio_type == 'para':
        new_human_speech1 = human_speech_array1
        new_human_speech2 = human_speech_array2
    elif audio_type == 'add':
        new_human_speech1 = np.concatenate([human_speech_array1, np.zeros(human_speech_array2.shape[0])])
        new_human_speech2 = np.concatenate([np.zeros(human_speech_array1.shape[0]), human_speech_array2])

    sum_human_speechs = new_human_speech1 + new_human_speech2
    return new_human_speech1, new_human_speech2, sum_human_speechs


# ---------------- Service Class ---------------- #

class InfiniteTalkService:
    def __init__(self, config_dict, model_path="/sharedata/checkpoints/ai_singer"):
        """
        初始化服务。
        config_dict: 包含所有模型配置和路径的字典 (对应原脚本的 args)
        """
        # 1. 配置参数转换
        self.model_path = model_path
        self.args = SimpleNamespace(**config_dict)
        self._validate_config()

        # 2. 分布式环境初始化
        self.rank = int(os.getenv("RANK", 0))
        self.world_size = int(os.getenv("WORLD_SIZE", 1))
        self.local_rank = int(os.getenv("LOCAL_RANK", 0))
        self.device = self.local_rank

        self._init_logging()
        self._init_distributed()

        # 3. 初始化 Wan Pipeline (T2V/I2V Model)
        # 新加入的逻辑
        if dist.is_initialized():
            base_seed = [self.args.base_seed] if self.rank == 0 else [None]
            dist.broadcast_object_list(base_seed, src=0)
            self.args.base_seed = base_seed[0]

        logging.info("Initializing InfiniteTalk Pipeline...")
        self.wan_i2v = wan.InfiniteTalkPipeline(
            config=WAN_CONFIGS[self.args.task],
            checkpoint_dir=self.args.ckpt_dir,
            quant_dir=self.args.quant_dir,
            device_id=self.device,
            rank=self.rank,
            t5_fsdp=self.args.t5_fsdp,
            dit_fsdp=self.args.dit_fsdp,
            use_usp=(self.args.ulysses_size > 1 or self.args.ring_size > 1),
            t5_cpu=self.args.t5_cpu,
            lora_dir=self.args.lora_dir,
            lora_scales=self.args.lora_scale,
            quant=self.args.quant,
            dit_path=self.args.dit_path,
            infinitetalk_dir=self.args.infinitetalk_dir,
            init_on_cpu=False
        )

        # 显存管理
        if self.args.num_persistent_param_in_dit is not None:
            self.wan_i2v.vram_management = True
            self.wan_i2v.enable_vram_management(
                num_persistent_param_in_dit=self.args.num_persistent_param_in_dit
            )

        # 4. 初始化 Audio Models (Wav2Vec)
        logging.info("Initializing Audio Models...")
        self.wav2vec_feature_extractor, self.audio_encoder = self._init_audio_models()

        # 5. 初始化 Super Resolution Model (AdcSR)
        # 注意：SR模型通常只需要在主进程或者负责保存的进程初始化，或者每张卡都跑
        # 原脚本逻辑是在 save_video_ffmpeg_sr 里面调用的，且 hardcode 了 device="cuda:0"
        # 这里为了支持多卡，我们让它在各自的 device 上初始化 (如果库支持的话)，或者只在 rank 0 初始化
        self.sr_app = None
        if self.rank == 0:
            logging.info("Initializing Super Resolution Model...")
            adcsr_dir = Path(self.args.adcsr_dir) # 从配置读取，不再硬编码
            # 假设 SR 模型可以在指定 device 运行
            # 如果 sr_pipe_trt 只能跑在 cuda:0，则需注意多进程冲突
            self.sr_app = VideoSuperResolution(adcsr_dir, device=f"cuda:{self.device}")
        self._warm_up()
    
    def _warm_up(self):
        print("=============================== 开始预热 ===============================")
        t0 = time.time()
        warm_up_data = [
            ['/sharedata/data/ai_singer/i2i_20251030/image/常回家看看.png','/sharedata/data/ai_singer/i2i_20251030/music/常回家看看.wav'], #13s
            ['/sharedata/data/ai_singer/i2i_20251030/image/红昭愿.png','/sharedata/data/ai_singer/i2i_20251030/music/红昭愿.wav'], #9s
            ['/sharedata/data/ai_singer/i2i_20251030/image/无名的人.png','/sharedata/data/ai_singer/i2i_20251030/music/无名的人.wav'], # 10s
        ]
        for idx,data in enumerate(warm_up_data):
            if os.path.exists(data[0]) and os.path.exists(data[1]):
                formatted_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
                save_dir = os.path.join("/sharedata/data/ai_singer", "tmp_warm_up", formatted_time)
                t1=time.time()
                self.generate(
                    prompt="人物在演唱歌曲，面带微笑，眼神专注，目视前方，肢体动作缓慢自然。镜头以人物为中心慢慢左右平移运动，人物背景里灯光动态变化柔和。",
                    image_path=data[0],
                    audio_path=data[1],
                    audio_vocal_path=data[1],
                    save_dir=save_dir,
                    height=854,
                )
                print(f"==> 第{idx+1}/{len(warm_up_data)}预热完成，耗时：{time.time() - t1:.2f}s")
                if self.rank==0 and os.path.exists(save_dir):
                    shutil.rmtree(save_dir)
                    #if dist.is_initialized() and self.world_size > 1: dist.barrier()
            else:
                print(f"==> 预热数据不存在：{data}")
        print(f"=============================== 预热完成，耗时：{time.time() - t0:.2f}s ===============================")

    def _validate_config(self):
        # 默认参数补全，防止 AttributeError
        defaults = {
            "task": "infinitetalk-14B",
            "size": "infinitetalk-480",
            "ckpt_dir": os.path.join(self.model_path, "Wan2.1-I2V-14B-480P"),
            "infinitetalk_dir": os.path.join(self.model_path, "Wan2_1-InfiniTetalk-Single_fp16.safetensors"),
            "wav2vec_dir": os.path.join(self.model_path, "chinese-wav2vec2-base"),
            "lora_dir": [os.path.join(self.model_path, "Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors")],
            "dit_path": os.path.join(self.model_path, "aniWan2114BFp8E4m3fn_i2v480pNew.safetensors"), 
            "adcsr_dir": self.model_path,
            "lora_scale": [1.0],
            "quant": None,
            "t5_fsdp": True, #False,
            "dit_fsdp": True, #False,
            "t5_cpu": False,
            "quant_dir": None,
            "sample_steps": 4, 
            "offload_model": False,
            "ulysses_size": 4,
            "ring_size": 1,
            "mode": "streaming",
            "base_seed": 666,
            "motion_frame": 9,
            "sample_shift": 11,
            "frame_num": 81, #97,
            "max_frame_num": 999999,
            "sample_text_guide_scale": 1.0,
            "sample_audio_guide_scale": 1.0,
            "color_correction_strength": 0.0,
            "num_persistent_param_in_dit": None,
            "audio_mode": "localfile",
            "use_teacache": False,
            "teacache_thresh": 0.2,
            "use_apg": False,
            "apg_momentum": -0.75,
            "apg_norm_threshold": 55,
            "scene_seg": False,
        }
        for k, v in defaults.items():
            if not hasattr(self.args, k):
                setattr(self.args, k, v)

    def _init_logging(self):
        if self.rank == 0:
            logging.basicConfig(
                level=logging.INFO,
                format="[%(asctime)s] %(levelname)s: %(message)s",
                handlers=[logging.StreamHandler(stream=sys.stdout)])
        else:
            logging.basicConfig(level=logging.ERROR)

    def _init_distributed(self):
        if self.world_size > 1:
            if not dist.is_initialized():
                torch.cuda.set_device(self.local_rank)
                dist.init_process_group(
                    backend="nccl",
                    init_method="env://",
                    rank=self.rank,
                    world_size=self.world_size)

            # Context Parallel Init (xfuser)
            if self.args.ulysses_size > 1 or self.args.ring_size > 1:
                from xfuser.core.distributed import init_distributed_environment, initialize_model_parallel
                init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())
                initialize_model_parallel(
                    sequence_parallel_degree=dist.get_world_size(),
                    ring_degree=self.args.ring_size,
                    ulysses_degree=self.args.ulysses_size,
                )

    def _init_audio_models(self):
        # 放到 CPU 还是 GPU 取决于显存，原脚本放在 custom_init 参数控制，默认 cpu
        # 为了速度，如果是推理服务，建议放到 device (GPU)
        device = 'cpu' # 原脚本逻辑倾向于 CPU 处理 feature extraction
        audio_encoder = Wav2Vec2Model.from_pretrained(self.args.wav2vec_dir, local_files_only=True).to(device)
        audio_encoder.feature_extractor._freeze_parameters()
        wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.args.wav2vec_dir, local_files_only=True)
        return wav2vec_feature_extractor, audio_encoder

    def _get_embedding(self, speech_array):
        # 内部 Helper：获取音频 Embedding
        sr = 16000
        device = 'cpu' # 对应 _init_audio_models 的 device

        audio_duration = len(speech_array) / sr
        video_length = audio_duration * 25

        audio_feature = np.squeeze(
            self.wav2vec_feature_extractor(speech_array, sampling_rate=sr).input_values
        )
        audio_feature = torch.from_numpy(audio_feature).float().to(device=device)
        audio_feature = audio_feature.unsqueeze(0)

        with torch.no_grad():
            embeddings = self.audio_encoder(audio_feature, seq_len=int(video_length), output_hidden_states=True)

        if len(embeddings) == 0:
            return None

        audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
        audio_emb = rearrange(audio_emb, "b s d -> s b d")
        audio_emb = audio_emb.cpu().detach()
        return audio_emb
    
    def _get_audio_emb(self, audio_data, sr=16000):
        if isinstance(audio_data, str):
            if not os.path.exists(audio_data):
                raise ValueError(f"Audio file {audio_data} does not exist.")
            audio_data, sr = librosa.load(audio_data, sr=sr)
        # if isinstance(audio_data, dict):
        #     sr = audio_data["sample_rate"]   
        #     audio_data = audio_data["waveform"][0]
        # if isinstance(audio_data, torch.Tensor):
        #     audio_data = audio_data.cpu().numpy()
        # print(f"==> [输入音频] {audio_data.shape}, {audio_data}")
        try:
            human_speech_array = loudness_norm(audio_data, sr)
        except Exception as e:
            human_speech_array = audio_data
            print("Warnning: Error in loudness_norm(), use auido directly.")
        emb = self._get_embedding(human_speech_array)
        audio_duration = librosa.get_duration(y=human_speech_array, sr=sr)
        return emb, audio_duration
    def generate(self,
                 prompt: str,
                 image_path: str,
                 audio_path: str,
                 save_dir: str = "./outputs", # 每个请求的临时存储目录必须不一样，否则会冲突，使用完需要工程侧清理
                 height = 854, 
                 width = 480,
                 audio_vocal_path = None, # 干声
                 start_time = None, # 支持音频裁剪
                 end_time = None, # 支持音频裁剪
                 use_dynamic_window = True, # 默认使用动态窗口
        ):
        print(f"==> [输入参数] prompt: {prompt}, image_path: {image_path}, audio_path: {audio_path}, save_dir: {save_dir}, height: {height}, width: {width}, audio_vocal_path: {audio_vocal_path}, start_time: {start_time}, end_time: {end_time}, use_dynamic_window: {use_dynamic_window}")
        audio_path2 = None
        bbox = None
        run_start_time = time.time()
        video_path = image_path
        audio_path1 = audio_path
        vocal_path1 = audio_vocal_path
        vocal_path2 = None

        # 1. 准备目录(注意命名不能用时间戳，要保证每张卡都能访问到同一目录)
        # task_id = os.path.basename(video_path).rsplit('.', 1)[0]
        task_id = ""
        current_save_dir = os.path.join(save_dir, task_id)
        audio_temp_dir = os.path.join(current_save_dir, "audio_temp")
        os.makedirs(audio_temp_dir, exist_ok=True)

        # 2. 裁剪音频
        if start_time is not None and len(start_time) and end_time is not None and len(end_time):
            print(f"==> [需要裁剪音频] 开始裁剪音频，裁剪时间段：{start_time} - {end_time}")
            audio_path_cropped = os.path.join(audio_temp_dir, "cropped_1.wav")
            success = simple_cut_audio(audio_path, start_time, end_time, audio_path_cropped)
            if success:
                audio_path1 = audio_path_cropped
            else:
                audio_path1 = audio_path
                print(f"==> [裁剪音频] 音频裁剪失败，使用原始音频文件：{audio_path}")
            print(f"==> [裁剪音频] 音频裁剪成功，裁剪后的音频文件为：{audio_path_cropped}")

            # 裁剪干声
            if audio_vocal_path is not None and len(audio_vocal_path):
                vocal_path_cropped = os.path.join(audio_temp_dir, "vocal_cropped_1.wav")
                success = simple_cut_audio(audio_vocal_path, start_time, end_time, vocal_path_cropped)
                if success:
                    vocal_path1 = vocal_path_cropped
                else:
                    vocal_path1 = audio_vocal_path
                    print(f"==> [裁剪干声] 音频裁剪失败，使用原始音频文件：{audio_vocal_path}")
            
            if audio_path2 is not None:
                audio_path2_cropped = os.path.join(audio_temp_dir, "cropped_2.wav")
                success = simple_cut_audio(audio_path2, start_time, end_time, audio_path2_cropped)
                if success:
                    audio_path2 = audio_path2_cropped
                else:
                    audio_path2 = audio_path2
                    print(f"==> [裁剪音频2] 音频裁剪失败，使用原始音频文件：{audio_path2}")
                print(f"==> [裁剪音频2] 音频裁剪成功，裁剪后的音频文件为：{audio_path2_cropped}")


        # 3. 处理混合音频 (用于最终合成视频)
        # if audio_path2:
        #     _, _, sum_human_speechs = audio_prepare_multi(audio_path1, audio_path2, 'add') # 默认叠加模式
        # else:
        #     sum_human_speechs = audio_prepare_single(audio_path1)

        # final_audio_path = os.path.join(audio_temp_dir, 'final_mix.wav')
        # sf.write(final_audio_path, sum_human_speechs, 16000)

        final_audio_path = audio_path1

        # 4. 人声分离+音频特征提取
        if vocal_path1 is None:
            try:
                vocal_path1 = final_audio_path
                vocal_data = vocal_separate(final_audio_path)
                # print("==> 人声分离成功")
                # emb, audio_duration = self._get_audio_emb(vocal_data)
                vocal_path_tmp = os.path.join(audio_temp_dir, 'vocal.wav')
                if self.rank == 0:
                    save_audio_execute(vocal_path_tmp, vocal_data)
                if self.world_size > 1: dist.barrier()
                if os.path.exists(vocal_path_tmp):
                    print("==> 人声分离成功，vocal_path_tmp:", vocal_path_tmp)
                    vocal_path1 = vocal_path_tmp
            except Exception as e:
                print(f"==> 人声分离失败，使用原始音频驱动，错误信息：{str(e)}")
                vocal_path1 = audio_path1
        
        # 音频 Embedding 提取
        if vocal_path1 is None or not os.path.exists(vocal_path1):
            vocal_path1 = audio_path1
        emb, audio_duration = self._get_audio_emb(vocal_path1)
        if dist.is_initialized() and self.world_size > 1: dist.barrier()
        # 5. 准备输入数据结构 (模拟原脚本的 input_json 解析后的结构)
        conds_list = []
        conds_list.append(video_path)
        conds_list.append(vocal_path1)
        if vocal_path2:
            conds_list.append(vocal_path2)


        # 6. 只处理第一个片段
        generated_video_chunks = []
        idx = 0
        input_clip = {
            'prompt': prompt,
            'cond_video': conds_list[0],
            'bbox': bbox,
            'video_audio': conds_list[1], # 指向混合好的音频
            'embedding':emb.cpu(),
            'audio_duration': audio_duration,
        }

        # cond_audio = {}
        # # --- Audio Embedding Extraction ---
        # if audio_path2:
        #     # Dual Audio
        #     s1, s2, _ = audio_prepare_multi(conds_list[1], conds_list[2], 'add')
        #     emb1 = self._get_embedding(s1)
        #     emb2 = self._get_embedding(s2)

        #     # 保存临时pt文件 (原脚本逻辑依赖文件路径，这里为了兼容保持保存文件逻辑，
        #     # 但更优的做法是修改 wan_i2v 接受 tensor，此处暂且按文件路径传递)
        #     emb1_path = os.path.join(audio_temp_dir, f'chunk_{idx}_1.pt')
        #     emb2_path = os.path.join(audio_temp_dir, f'chunk_{idx}_2.pt')

        #     if self.rank == 0:
        #         torch.save(emb1, emb1_path)
        #         torch.save(emb2, emb2_path)

        #     # Barrier ensuring file exists for all ranks
        #     if self.world_size > 1: dist.barrier()

        #     cond_audio['person1'] = emb1_path
        #     cond_audio['person2'] = emb2_path
        # else:
        #     # Single Audio
        #     s1 = audio_prepare_single(conds_list[1])
        #     emb1 = self._get_embedding(s1)
        #     # print("==> 提取的音频特征:", emb1.shape, emb1)
        #     emb1_path = os.path.join(audio_temp_dir, f'chunk_{idx}_1.pt')

        #     if self.rank == 0:
        #         torch.save(emb1, emb1_path)

        #     if self.world_size > 1: dist.barrier()

        #     cond_audio['person1'] = emb1_path
        #     if not os.path.exists(emb1_path):
        #         raise ValueError(f"Embedding file {emb1_path} does not exist.")

        input_clip['cond_audio'] = dict()

        # --- Generation ---
        logging.info(f"Generating chunk {idx}...")

        # 7. 处理动态窗口逻辑，根据音频时长自适应调整window size，避免最后一段推理太多无用帧，浪费推理时间，变化范围：81~100
        motion_frame = 9 # 每个窗口重叠的帧数
        dynamic_window_size = self.args.frame_num
        # y, sr = librosa.load(vocal_path1, sr=16000)
        # audio_duration = librosa.get_duration(y=y, sr=sr) # s
        if use_dynamic_window:
            # 每个窗口都有重叠，其实真正的推理size为 dynamic_window_size - motion_frame
            run_time, run_time_res = divmod(motion_frame + 25 * audio_duration, dynamic_window_size - motion_frame)
            if run_time > 0 and run_time_res > 0:
                window_acc = int(round(run_time_res / run_time + 0.5))
                if window_acc <= 20:
                    # 调整size，可减少一次采样
                    dynamic_window_size += window_acc
                    # 向上调整到4的倍数+1
                    dynamic_window_size = round((dynamic_window_size - 1) /4 + 0.5) * 4 + 1
                else:
                    # 不调整size
                    run_time += 1
            print(f"==> 音频时长：{audio_duration} s, 调整后的动态window_size: {dynamic_window_size}, 推理次数: {run_time}，最后一段多余帧数: {run_time_res}")

        # 8. 调用 Wan Pipeline+ Infinitetalk生成视频
        # print("==> 输入input_clip:",input_clip)
        video_tensor = self.wan_i2v.generate_infinitetalk(
            input_clip,
            n_prompt="明亮的色调、曝光过度、静止、模糊的细节、字幕、风格、作品、画作、图像、静止的物体、整体呈灰色、质量最差、质量低下、JPEG 压缩残留、难看、不完整、多出的手指、画得糟糕的手部、画得糟糕的面部、变形、畸形、形状不规则的肢体、融合的手指、静态图片、杂乱的背景、三只腿、闭眼、背景中有许多人物、模糊、失焦、低分辨率、过度曝光、欠曝、噪点、多余的肢体",
            size_buckget=self.args.size,
            motion_frame=motion_frame,
            frame_num=dynamic_window_size,
            shift=self.args.sample_shift,
            sampling_steps=self.args.sample_steps,
            text_guide_scale=self.args.sample_text_guide_scale,
            audio_guide_scale=self.args.sample_audio_guide_scale,
            seed=self.args.base_seed,
            offload_model=self.args.offload_model,
            max_frames_num=dynamic_window_size if self.args.mode == 'clip' else self.args.max_frame_num,
            color_correction_strength=self.args.color_correction_strength,
            extra_args=self.args,
            target_h=height, 
            target_w=width
        )
        # print("video_tensor.shape：",video_tensor.shape)
        generated_video_chunks.append(video_tensor)
        print("==> 总体耗时（不带超分）：",time.time()-run_start_time,'秒')

        # 9. 后处理与保存
        final_output = None
        if self.rank == 0:
            sum_video = torch.cat(generated_video_chunks, dim=1)
            formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"res_{formatted_time}.mp4"
            raw_save_path = os.path.join(current_save_dir, filename)

            # 使用 SR 模型增强并保存
            logging.info("Running Super Resolution and Saving...")
            if self.sr_app:
                # 调用 SR 模型 (save_video_ffmpeg_sr 逻辑内联化)
                t1 = time.time()
                # enhance_video 通常会保存文件并返回路径
                res_path = self.sr_app.enhance_video(
                    sum_video,
                    raw_save_path, # 这里可能是 prefix 或者 full path，视 sr_app 实现而定
                    keep_audio=True,
                    audio_raw_path=final_audio_path,
                    batch_size=4
                )
                print(f"Super Resolution Time: {time.time() - t1:.2f}s")
                final_output = res_path
            else:
                # Fallback without SR
                save_video_ffmpeg(sum_video, raw_save_path, [final_audio_path], fps=25)
                # Mux audio
                # (Need ffmpeg mux logic here if SR doesn't do it)
                final_output = {
                    "output_mp4_path": raw_save_path,
                    "output_cover_path": "",
                }

        # 10. 等待主进程完成保存
        if dist.is_initialized() and self.world_size > 1: dist.barrier()
        end_time = time.time()
        process_speed = None
        if audio_duration is not None and audio_duration > 0:
            process_speed =  (time.time() - run_start_time) / audio_duration
        print(f"==> Pipiline总体耗时: {end_time - run_start_time}s, 每秒音频处理耗时: {process_speed}s， 音频时长: {audio_duration}s")
        
        return final_output if self.rank == 0 else None

# ---------------- Test / Verification Block ---------------- #

if __name__ == "__main__":
    ## 启动命令示例
    """
    export CUDA_VISIBLE_DEVICES='0,1,2,3'
    GPU_NUM=4
    torchrun --nproc_per_node=$GPU_NUM --standalone infinitetalk_parallel_pipeline.py --ulysses_size=$GPU_NUM
    """

    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    args = parser.parse_args()

    # 模拟外部配置
    config = {
        "ulysses_size": args.ulysses_size, # 模型并行度，GPU数量
    }

    print(">>> Initializing Service...")
    # 1. 实例化服务 (模型仅加载一次)
    service = InfiniteTalkService(config)
    print(">>> Service Initialized. Ready for inference.")

    # Load your input data
    with open('examples/1.json', 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    print(">>> Input data loaded.", input_data)

    # 模拟
    json_path = "/sharedata/user/chengkaichang/projects/data/ai_singer/jiutian_dev/infinitetalk_test_data.json"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    parent_dir = os.path.dirname(json_path)
    cnt = 0
    for k,v in data.items():
        image_folder = v["image_folder"]
        image_folder = os.path.join(parent_dir, image_folder)
        audio_path = v["audio_path"]
        audio_path = os.path.join(parent_dir, audio_path)
        mv_prompt = v["mv_prompt"]
        images = glob.glob(image_folder + '/*.jpg') + glob.glob(image_folder + '/*.png') + glob.glob(image_folder + '/*.webp')
        for image_path in images:
    
            #image_path = input_data["cond_video"]
            #audio_path = input_data["cond_audio"]["person1"]
            prompt = mv_prompt
            if os.path.exists(image_path) and os.path.exists(audio_path):
                print(">>> Starting Generation ...")

                # 验证模型只需加载一次：再次调用 generate
                print(">>> Starting Generation  (Reuse loaded models)...")
                t1=time.time()
                result2 = service.generate(
                    prompt=prompt,
                    image_path=image_path,
                    audio_path=audio_path,
                    audio_vocal_path=audio_path,
                    save_dir="./gpu4_driver_580_fsdp"
                )
                cnt += 1
                print(f"{cnt}总耗时：", time.time() - t1,"秒")
                print(f">>> Generation  Result: {result2}")
            else:
                print("Note: Test files not found. Creating dummy test logic passed.")

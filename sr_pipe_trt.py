import os
import os.path as osp
import torch
import cv2
import glob
import subprocess
import numpy as np
import time
from tqdm import tqdm
import tensorrt as trt
import torch.nn.functional as F
import threading
import queue


# ===========================
# ffmpeg 写线程：边超分边写视频（支持 NVENC）
# ===========================
class FFmpegWriterThread(threading.Thread):
    def __init__(
        self,
        width,
        height,
        fps,
        input_video_path,
        output_video_path,
        keep_audio=True,
        audio_raw_path=None,
        use_nvenc=False,
        max_queue_size=64,
    ):
        super().__init__(daemon=True)
        self.width = width
        self.height = height
        self.fps = fps
        self.input_video_path = input_video_path
        self.output_video_path = output_video_path
        self.keep_audio = keep_audio
        self.audio_raw_path = audio_raw_path
        self.use_nvenc = use_nvenc

        self.queue = queue.Queue(maxsize=max_queue_size)
        self._stop_event = threading.Event()
        self._proc = None

    def run(self):
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",  # 输入 RGB
            "-s", f"{self.width}x{self.height}",
            "-r", str(self.fps),
            "-i", "-",            # stdin: 视频流
        ]

        if self.keep_audio:
            audio_src = None
            if self.audio_raw_path is not None and os.path.exists(self.audio_raw_path):
                audio_src = self.audio_raw_path
            else:
                audio_src = self.input_video_path
            cmd += [
                "-i", audio_src,          # 第 2 路输入：音频
                "-map", "0:v:0",
                "-map", "1:a:0?",
                "-c:a", "aac",
                "-strict", "experimental",
            ]
        else:
            cmd += ["-an"]               # 不要音频

        if self.use_nvenc:
            # GPU 编码
            cmd += [
                "-c:v", "h264_nvenc",
                "-preset", "p4",         # 可改 p3/p5 做画质-速度平衡
                "-rc", "vbr",
                "-cq", "23",
                "-b:v", "0",
            ]
        else:
            # CPU x264 编码
            cmd += [
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", "23",
            ]

        cmd += [
            "-pix_fmt", "yuv420p",
            self.output_video_path,
        ]

        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

        while True:
            frame = self.queue.get()
            if frame is None:   # 结束标记
                break
            # frame: (H,W,3) RGB uint8
            self._proc.stdin.write(frame.tobytes())
            self.queue.task_done()

        # 收尾
        try:
            self._proc.stdin.close()
        except Exception:
            pass
        self._proc.wait()

    def write(self, frame: np.ndarray):
        """向队列写一帧 (H,W,3) RGB uint8。"""
        self.queue.put(frame)

    def close(self):
        """发送结束标记并等待线程退出。"""
        self.queue.put(None)
        self.join()


# ===========================
# TensorRT RealESRGAN 封装（TensorRT 10 + 动态 batch 1~8）
# ===========================
class TrtRealESRGAN:
    """
    使用 TensorRT 10 引擎进行 RealESRGAN 超分。

    约定：
      - 引擎输入: NCHW, RGB, [0,1]，dtype 可能是 fp32 或 fp16（看 engine 配置）
      - 输出:     NCHW, RGB, [0,1]，dtype 同上
      - 分辨率 H,W 固定 (例如 540x960)
      - batch 维动态：1~8（通过 min/opt/maxShapes 配出来）

    不使用 pycuda，直接用 torch.Tensor.data_ptr 提供 GPU 指针。
    """

    def __init__(self, engine_path: str, device: torch.device, max_batch: int = 8):
        assert osp.exists(engine_path), f"TensorRT engine 不存在: {engine_path}"
        self.engine_path = engine_path
        self.device = device
        self.max_batch = max_batch

        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f:
            engine_bytes = f.read()

        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        assert self.engine is not None, "反序列化 TensorRT engine 失败"

        self.context = self.engine.create_execution_context()
        assert self.context is not None, "创建 TensorRT execution context 失败"

        # === TensorRT 10: 用 num_io_tensors / get_tensor_name / get_tensor_mode ===
        self.input_name = None
        self.output_name = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
            elif mode == trt.TensorIOMode.OUTPUT:
                self.output_name = name

        assert self.input_name is not None, "未找到输入 tensor"
        assert self.output_name is not None, "未找到输出 tensor"

        # 记录 TensorRT dtype -> 对应 torch dtype
        self.input_dtype_trt = self.engine.get_tensor_dtype(self.input_name)
        self.output_dtype_trt = self.engine.get_tensor_dtype(self.output_name)

        def _to_torch_dtype(dt):
            if dt == trt.DataType.FLOAT:
                return torch.float32
            elif dt == trt.DataType.HALF:
                return torch.float16
            else:
                raise RuntimeError(f"暂不支持的 TRT dtype: {dt}")

        self.torch_input_dtype = _to_torch_dtype(self.input_dtype_trt)
        self.torch_output_dtype = _to_torch_dtype(self.output_dtype_trt)

        # 获取 tensor shape（batch 维可能是 -1，H,W 是固定正数）
        input_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        output_shape = tuple(self.engine.get_tensor_shape(self.output_name))
        self._raw_input_shape = input_shape
        self._raw_output_shape = output_shape

        # input_shape 可能是 (-1,3,540,960) 或 (1,3,540,960)
        self.in_h = input_shape[-2] if input_shape[-2] > 0 else None
        self.in_w = input_shape[-1] if input_shape[-1] > 0 else None

        self.out_h = output_shape[-2] if output_shape[-2] > 0 else None
        self.out_w = output_shape[-1] if output_shape[-1] > 0 else None

        if self.in_h is not None and self.out_h is not None:
            self.scale = self.out_h // self.in_h
        else:
            self.scale = None

        print(f"==> TensorRT RealESRGAN engine: {engine_path}")
        print(f"    Input  tensor: {self.input_name}, shape={input_shape}, dtype={self.input_dtype_trt}")
        print(f"    Output tensor: {self.output_name}, shape={output_shape}, dtype={self.output_dtype_trt}")
        if self.in_h is not None and self.in_w is not None:
            print(f"    Input  HxW: {self.in_h}x{self.in_w}")
        if self.out_h is not None and self.out_w is not None:
            print(f"    Output HxW: {self.out_h}x{self.out_w}")
        print(f"    Scale factor: x{self.scale}")
        print(f"    Torch input dtype : {self.torch_input_dtype}")
        print(f"    Torch output dtype: {self.torch_output_dtype}")

    @property
    def input_size_hw(self):
        """返回 (H, W)，方便外部做 resize。"""
        if self.in_h is None or self.in_w is None:
            raise RuntimeError(
                f"引擎输入分辨率是动态的：raw shape={self._raw_input_shape}，"
                f"当前代码假设 H,W 固定，请检查 ONNX/TRT 构建。"
            )
        return self.in_h, self.in_w

    def infer(self, batch_tensor: torch.Tensor) -> torch.Tensor:
        """
        batch_tensor: torch.Tensor, (B,3,H,W), 在 CPU 或 GPU 上均可,
                      RGB, [0,1], dtype 允许和 engine 不同，会自动转换成匹配 dtype。
        返回: torch.Tensor, (B,3,H_out,W_out), 在 GPU 上，dtype = engine 的输出 dtype。
        """
        assert batch_tensor.ndim == 4 and batch_tensor.shape[1] == 3, \
            f"期望输入 (B,3,H,W)，实际 {batch_tensor.shape}"

        B, _, H, W = batch_tensor.shape
        assert 1 <= B <= self.max_batch, f"batch 大小 {B} 超出引擎支持范围 1~{self.max_batch}"

        # 检查分辨率
        if self.in_h is not None and self.in_w is not None:
            assert H == self.in_h and W == self.in_w, \
                f"输入尺寸与引擎不匹配: 引擎({self.in_h},{self.in_w}) vs 当前({H},{W})"

        # 拷到 GPU，并转成 TRT 要求的 dtype
        if batch_tensor.device != self.device or batch_tensor.dtype != self.torch_input_dtype:
            inp_gpu = batch_tensor.to(self.device, dtype=self.torch_input_dtype, non_blocking=True).contiguous()
        else:
            inp_gpu = batch_tensor.contiguous()

        # 设置真实输入 shape（包括 batch 维）
        ok = self.context.set_input_shape(self.input_name, tuple(inp_gpu.shape))
        assert ok, f"set_input_shape 失败，shape={tuple(inp_gpu.shape)}"

        # 根据当前 context 获取输出 shape（会带上真实 batch 维）
        out_shape = tuple(self.context.get_tensor_shape(self.output_name))  # e.g. (B,3,H_out,W_out)
        out_gpu = torch.empty(out_shape, dtype=self.torch_output_dtype, device=self.device)

        # 绑定指针
        self.context.set_tensor_address(self.input_name, int(inp_gpu.data_ptr()))
        self.context.set_tensor_address(self.output_name, int(out_gpu.data_ptr()))

        # 使用当前 PyTorch stream
        stream = torch.cuda.current_stream(self.device)
        self.context.execute_async_v3(stream_handle=stream.cuda_stream)
        stream.synchronize()

        # 返回 GPU Tensor，由上层决定何时搬回 CPU
        return out_gpu


# ============================================
# 使用 TRT 的视频超分主类（真正 batch 处理 + 流式写视频）
# ============================================
class VideoSuperResolution:
    def __init__(
        self,
        config_dir="/sharedata/user/duzongcai/repos/AdcSR_server/models",
        device="cuda:0",
        scale=2,
        use_nvenc=False,   # 🔥 是否启用 NVENC
    ):
        self.device = torch.device(device)
        self.scale = scale
        self.use_nvenc = use_nvenc

        # 选择 TensorRT 引擎文件（batch<=8 FP16 engine）
        if self.scale == 2:
            engine_name = "realesrgan_x2_960x540_b8_fp16_io.plan"
            # engine_name = "realesrgan_x2_960x540_b8_int8_new_2.plan"
        else:
            engine_name = "realesrgan_x4_b8_fp16.plan"

        engine_path = osp.join(config_dir, engine_name)
        self.model = TrtRealESRGAN(engine_path, device=self.device, max_batch=8)

        # 根据引擎的输入尺寸设置读取视频时的 resize 尺寸
        self.input_h, self.input_w = self.model.input_size_hw

        # 给 TRT 的输入统一用 engine 对应的 dtype（可能是 fp16 或 fp32）
        self.dtype = self.model.torch_input_dtype

    def read_video_frames(self, video_path):
        """
        读取视频或帧列表，并 resize 到 TensorRT 引擎期望的分辨率。
        返回:
          - torch.Tensor, shape (T,3,H,W), RGB,[0,1] (CPU, float32)
          - fps: 原视频帧率（读取失败则默认 25）
        """
        target_width = self.input_w
        target_height = self.input_h
        fps = 25.0

        if isinstance(video_path, torch.Tensor):
            return (video_path+1)*0.5, fps
        frames = []
        # 如果video_path是一个包含视频帧路径的list
        if isinstance(video_path, list):
            
            for frame_path in video_path:
                frame = cv2.imread(frame_path)
                frames.append(torch.from_numpy(frame))
        else:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(torch.from_numpy(frame))
            cap.release()

        if len(frames) == 0:
            return torch.empty(0, 3, target_height, target_width), fps

        return torch.stack(frames, dim=0), fps

    def crop_and_resize_image(self, image, target_width=1080, target_height=1920):
        height, width = image.shape[:2]
        target_ratio = target_width / target_height

        if width / height > target_ratio:
            new_width = int(height * target_ratio)
            new_height = height
            x_offset = (width - new_width) // 2
            y_offset = 0
        else:
            new_width = width
            new_height = int(width / target_ratio)
            x_offset = 0
            y_offset = (height - new_height) // 2

        cropped_image = image[y_offset:y_offset + new_height, x_offset:x_offset + new_width]
        resized_image = cv2.resize(cropped_image, (target_width, target_height), interpolation=cv2.INTER_AREA)
        return resized_image

    def center_crop_width(self, tensor, target_width=1080):
        _, _, h, w = tensor.shape
        start = (w - target_width) // 2
        return tensor[:, :, :, start:start + target_width]

    def extract_first_frame_cv(self, video_path, image_path):
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(image_path, frame)
        cap.release()

    def enhance_video(self, input_path, output_path, batch_size=8, keep_audio=True, audio_raw_path=None):
        """
        使用 TensorRT RealESRGAN 对视频做超分，真正 batch 推理 + 流式写 MP4。
        - batch_size 建议 <= 8，对应引擎 maxShapes。
        - 尾 batch 不足时复制最后一帧补齐到 batch_size，推理后再裁掉。
        - 后处理（*255/clamp/uint8/transpose）在 GPU 上做，尽量减少 CPU 开销。
        - 使用 FFmpegWriterThread 边超分边写视频。
        """
        output_mp4_path = ""
        output_cover_path = ""
        os.makedirs(osp.dirname(output_path), exist_ok=True)

        tensors, fps = self.read_video_frames(input_path)
        ## 统一预处理
        if tensors.shape[0] == 3:
            tensors = tensors.permute(1, 0, 2, 3).contiguous()
        elif tensors.shape[3] == 3:
            tensors = tensors.permute(0,3,1,2).contiguous()
        print("==>输入tensor shape:", tensors.shape)
        # 归一化到 0~1
        if tensors.max() > 1.0:
            tensors = tensors.float() / 255.0
            tensors = torch.flip(tensors, dims=[1])  # 在通道维度翻转，BGR -> RGB
            
        if tensors.shape[2] != 960 or tensors.shape[3] != 540:
            # 对于形状 [B, C, H, W] 的批量图像（PyTorch常用格式）
            tensors = F.interpolate(tensors, size=(960, 540), mode='bicubic', align_corners=False)
            tensors = tensors.clip_(0., 1.)
        
        num_frames = tensors.shape[0]
        if num_frames == 0:
            print(f"❌ 无法读取视频帧: {input_path}")
            return {"output_mp4_path": "", "output_cover_path": ""}

        t1 = time.time()

        writer = None  # ffmpeg 写线程，首次拿到超分结果后再创建
        total_written = 0

        for i in range(0, num_frames, batch_size):
            batch = tensors[i:i + batch_size]  # CPU, float32, (B,3,H,W)
            B = batch.shape[0]

            # 拷到 GPU，并转成 engine 要求的 dtype（fp16 / fp32）
            batch = batch.to(device=self.device, dtype=self.dtype, non_blocking=True)

            # 尾 batch 不足时复制最后一帧补齐到 batch_size
            if B < batch_size:
                pad = batch_size - B
                last = batch[-1:].repeat(pad, 1, 1, 1)
                batch_padded = torch.cat([batch, last], dim=0)
            else:
                batch_padded = batch

            # TensorRT 推理（一次跑整批），返回 GPU Tensor
            sr_gpu = self.model.infer(batch_padded)      # (batch_size,3,H_out,W_out), fp16/fp32
            sr_gpu = sr_gpu[:B]                          # 去掉补的帧，仅保留真实的 B 帧

            # 在 GPU 上做 *255 + clamp + uint8 + NCHW->NHWC
            if sr_gpu.dtype != torch.float32:
                sr_post = sr_gpu.to(torch.float32)
            else:
                sr_post = sr_gpu
            sr_post = (sr_post * 255.0).clamp(0, 255).to(torch.uint8)   # (B,3,H,W)
            sr_post = sr_post.permute(0, 2, 3, 1).contiguous()          # (B,H,W,3) RGB uint8

            # 一次性搬回 CPU
            sr = sr_post.cpu().numpy()                                  # (B,H,W,3)

            # 第一次拿到超分结果时，创建 ffmpeg 写线程
            if writer is None and B > 0:
                H_out, W_out = sr.shape[1:3]
                writer = FFmpegWriterThread(
                    width=W_out,
                    height=H_out,
                    fps=fps,
                    input_video_path=input_path,
                    output_video_path=output_path,
                    keep_audio=keep_audio,
                    audio_raw_path=audio_raw_path,
                    use_nvenc=self.use_nvenc,
                    max_queue_size=64,
                )
                writer.start()

            # 把当前 batch 的帧送入写线程
            if writer is not None:
                for f in sr:
                    writer.write(f)
                    total_written += 1

        t2 = time.time()

        # 结束写线程
        if writer is not None:
            writer.close()
            output_mp4_path = output_path
            t3 = time.time()
            print(f"==> 超分流程耗时：超分{num_frames}帧：{t2 - t1:.3f}秒，写MP4：{t3 - t2:.3f}秒（共写入{total_written}帧）")
            output_cover_path = output_path.rsplit(".", 1)[0] + ".png"
            self.extract_first_frame_cv(output_path, output_cover_path)
        else:
            print("❌ 无输出视频, %s" % input_path)

        res = {
            "output_mp4_path": output_mp4_path,
            "output_cover_path": output_cover_path,
        }
        return res

    def enhance_video_thread(self, *args, **kwargs):
        print("⚠ 当前版本建议使用单线程 enhance_video，避免多线程与 TensorRT 交互导致不稳定。")
        return self.enhance_video(*args, **kwargs)


if __name__ == "__main__":
    config_dir = "/sharedata/user/duzongcai/repos/AdcSR_server/models"
    input_dir = "/sharedata/user/chengkaichang/projects/data/ai_singer/i2v_batch1110_h100_432"
    output_dir = "/sharedata/user/chengkaichang/projects/data/ai_singer/i2v_batch1110_h100_432_trt_2"
    os.makedirs(output_dir, exist_ok=True)

    # 可以把 use_nvenc=False 试一下 CPU x264，对比写 MP4 耗时
    model = VideoSuperResolution(config_dir, device="cuda:0", scale=2, use_nvenc=False)

    log_path = osp.join(output_dir, "timing_log.txt")
    with open(log_path, "w", encoding="utf-8") as log_file:
        for video_path in tqdm(glob.glob(osp.join(input_dir, "*.mp4"))):
            basename = osp.basename(video_path)
            output_path = osp.join(output_dir, basename)
            start_time = time.time()
            torch.cuda.synchronize()
            res = model.enhance_video(video_path, output_path, keep_audio=True, batch_size=8)
            torch.cuda.synchronize()
            duration = time.time() - start_time
            print(f"✅ 超分完成: {res}，耗时：{duration:.2f} 秒")
            log_file.write(f"{basename}\t{duration:.2f} 秒\n")

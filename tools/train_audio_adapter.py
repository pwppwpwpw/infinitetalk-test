"""Training script for the InfiniteTalk audio adapter.

This script fine-tunes :class:`AudioProjModel` so that the audio
embeddings extracted in inference (see ``generate_infinitetalk.py``)
are mapped to conditioning tokens that match a teacher target.

Dataset format
--------------
Provide a JSON manifest where each element contains at least two
fields:

- ``"audio"``: path to a ``.pt`` file created by
  ``generate_infinitetalk.get_embedding`` (shape ``[T, L, C]`` where
  ``T`` is the number of frames, ``L`` is the wav2vec layer count, and
  ``C`` is the hidden size).
- ``"target"``: path to a ``.pt`` tensor with shape
  ``[T, context_tokens, output_dim]`` containing the teacher audio
  conditioning tokens for each frame.

Each JSON entry can optionally include ``"weight"`` to re-weight the
example loss.

The loader will automatically build the sliding audio windows expected
by ``AudioProjModel`` (matching the logic in ``wan/multitalk.py``) and
pad variable-length clips inside a batch.

Example usage
-------------
python tools/train_audio_adapter.py \
    --manifest /path/to/train.json \
    --output-dir outputs/audio_adapter \
    --batch-size 2 --epochs 5 --lr 1e-4
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from wan.modules.multitalk_model import AudioProjModel


def _load_tensor(path: str) -> torch.Tensor:
    tensor = torch.load(path)
    if not torch.is_tensor(tensor):
        raise ValueError(f"Expected a tensor at {path}, got {type(tensor)}")
    return tensor.float()


def _build_audio_windows(
    audio: torch.Tensor,
    audio_window: int,
    vae_scale: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert frame-wise audio embeddings into first/latter windows.

    Args:
        audio: Tensor with shape ``[T, S, C]`` where ``T`` is frames.
        audio_window: Number of tokens around the center frame (must be
            odd to build symmetric context).
        vae_scale: Temporal downsample factor used in the VAE; this
            controls how many audio windows are grouped for each
            diffusion step.

    Returns:
        first_frame_audio_emb_s: ``[1, audio_window, S, C]`` slice for
            the first frame.
        latter_frame_audio_emb_s: ``[N_t, seq_len_vf, S, C]`` tensor for
            the remaining frames after grouping.
    """
    if audio_window % 2 == 0:
        raise ValueError("audio_window must be odd to have a center index")

    num_frames, num_layers, hidden = audio.shape
    indices = torch.arange(audio_window, device=audio.device) - audio_window // 2
    center = torch.arange(num_frames, device=audio.device).unsqueeze(1)
    window_indices = torch.clamp(center + indices.unsqueeze(0), 0, num_frames - 1)
    windows = audio[window_indices]  # [T, audio_window, S, C]

    # First-frame slice used by the model
    first_frame_audio_emb_s = windows[:1]

    # Remaining frames are grouped by the VAE temporal stride
    latter = windows[1:]
    if latter.numel() == 0:
        # If we only have one frame, mirror the first frame to keep
        # shapes consistent with downstream code.
        latter = windows[:1].clone()

    # Pad so that length is divisible by vae_scale
    target_length = int(math.ceil(latter.shape[0] / vae_scale) * vae_scale)
    if target_length > latter.shape[0]:
        pad = latter[-1:].expand(target_length - latter.shape[0], *latter.shape[1:])
        latter = torch.cat([latter, pad], dim=0)

    latter = latter.view(-1, vae_scale, audio_window, num_layers, hidden)
    middle_index = audio_window // 2
    first_slice = latter[:, :1, : middle_index + 1]
    mid_slice = latter[:, 1:-1, middle_index:middle_index + 1]
    last_slice = latter[:, -1:, middle_index:]
    latter_frame_audio_emb_s = torch.cat([first_slice, mid_slice, last_slice], dim=1)

    return first_frame_audio_emb_s, latter_frame_audio_emb_s


@dataclass
class Batch:
    first: torch.Tensor
    latter: torch.Tensor
    target: torch.Tensor
    mask: torch.Tensor
    weight: torch.Tensor


class AudioAdapterDataset(Dataset):
    def __init__(self, manifest: str, audio_window: int, vae_scale: int):
        with open(manifest, "r") as f:
            self.entries: List[Dict] = json.load(f)
        self.audio_window = audio_window
        self.vae_scale = vae_scale

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        item = self.entries[idx]
        audio = _load_tensor(item["audio"])  # [T, L, C]
        target = _load_tensor(item["target"])  # [T, M, D]
        weight = float(item.get("weight", 1.0))

        first, latter = _build_audio_windows(audio, self.audio_window, self.vae_scale)
        return first, latter, target, weight


class _PadAndStack:
    def __init__(self, audio_window: int, context_tokens: int, output_dim: int):
        self.audio_window = audio_window
        self.context_tokens = context_tokens
        self.output_dim = output_dim

    def __call__(self, batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]]) -> Batch:
        max_frames = max(item[2].shape[0] for item in batch)
        first_list, latter_list, target_list, weights = [], [], [], []
        for first, latter, target, weight in batch:
            frames = target.shape[0]
            # Pad targets
            pad_frames = max_frames - frames
            if pad_frames > 0:
                pad_target = target.new_zeros(pad_frames, self.context_tokens, self.output_dim)
                target = torch.cat([target, pad_target], dim=0)

            # first: [1, W, L, C] -> keep shape
            # latter: [N_t, seq_len_vf, L, C] -> pad along first dim to align with frames-1
            expected_latter_frames = max(1, max_frames - 1)
            if latter.shape[0] < expected_latter_frames:
                pad_latter = latter[-1:].expand(expected_latter_frames - latter.shape[0], *latter.shape[1:])
                latter = torch.cat([latter, pad_latter], dim=0)
            first_list.append(first)
            latter_list.append(latter)
            target_list.append(target)
            weights.append(weight)

        first = torch.stack(first_list, dim=0)  # [B, 1, W, L, C]
        latter = torch.stack(latter_list, dim=0)  # [B, Nt, seq_len_vf, L, C]
        target = torch.stack(target_list, dim=0)  # [B, T, M, D]
        mask = torch.zeros_like(target[..., 0])  # [B, T, M]
        for i, (_, _, tgt, _) in enumerate(batch):
            mask[i, : tgt.shape[0], :] = 1
        weight = torch.tensor(weights, dtype=target.dtype).view(-1, 1, 1)
        return Batch(first=first, latter=latter, target=target, mask=mask, weight=weight)


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = AudioAdapterDataset(args.manifest, args.audio_window, args.vae_scale)
    collate = _PadAndStack(args.audio_window, args.context_tokens, args.output_dim)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )

    model = AudioProjModel(
        seq_len=args.audio_window,
        seq_len_vf=args.audio_window + args.vae_scale - 1,
        intermediate_dim=args.intermediate_dim,
        output_dim=args.output_dim,
        context_tokens=args.context_tokens,
        norm_output_audio=args.norm_output_audio,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision)
    os.makedirs(args.output_dir, exist_ok=True)

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            batch = Batch(
                first=batch.first.to(device),
                latter=batch.latter.to(device),
                target=batch.target.to(device),
                mask=batch.mask.to(device),
                weight=batch.weight.to(device),
            )

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=args.mixed_precision):
                pred = model(batch.first, batch.latter)
                # pred: [B, T, M, D]
                loss = torch.nn.functional.l1_loss(pred, batch.target, reduction="none")
                loss = (loss * batch.mask.unsqueeze(-1)).mean(dim=(1, 2, 3))
                loss = (loss * batch.weight.squeeze()).mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            global_step += 1

            if args.log_every and global_step % args.log_every == 0:
                print(f"Step {global_step}: loss={loss.item():.6f}")

        avg_loss = epoch_loss / max(len(dataloader), 1)
        print(f"Epoch {epoch+1} finished. Average L1 loss: {avg_loss:.6f}")

        if args.save_every and (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(args.output_dir, f"audio_adapter_epoch{epoch+1}.pt")
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "args": vars(args),
            }, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

    final_path = os.path.join(args.output_dir, "audio_adapter_final.pt")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": args.epochs,
        "args": vars(args),
    }, final_path)
    print(f"Training complete. Final weights saved to {final_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the InfiniteTalk audio adapter")
    parser.add_argument("--manifest", type=str, required=True, help="Path to JSON manifest with training pairs")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to store checkpoints")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log-every", type=int, default=50, help="Log every N steps (0 disables logging)")
    parser.add_argument("--save-every", type=int, default=1, help="Save checkpoint every N epochs")
    parser.add_argument("--audio-window", type=int, default=5, help="Sliding window size used during inference")
    parser.add_argument("--vae-scale", type=int, default=4, help="Temporal VAE scale used to group frames")
    parser.add_argument("--context-tokens", type=int, default=32, help="Number of context tokens emitted by the adapter")
    parser.add_argument("--output-dim", type=int, default=768, help="Dimensionality of context tokens")
    parser.add_argument("--intermediate-dim", type=int, default=512, help="Hidden size inside the adapter MLP")
    parser.add_argument("--norm-output-audio", action="store_true", help="Apply LayerNorm to adapter outputs")
    parser.add_argument("--mixed-precision", action="store_true", help="Enable AMP for faster training")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)

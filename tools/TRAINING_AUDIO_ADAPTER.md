# Training the Audio Adapter

This guide explains what you need to do to fine-tune the audio adapter (`AudioProjModel`) using `tools/train_audio_adapter.py`, including how to organize the dataset.

## What you need to prepare (besides the dataset)

1. **Environment**
   - Install the project requirements (GPU + CUDA recommended):
     ```bash
     pip install -r requirements.txt
     ```
   - Ensure PyTorch can see your GPU if you plan to train with `--device cuda`.

2. **Checkpoints and configs**
   - Keep the InfiniteTalk weights and configs available so you can reuse the same audio pre-processing pipeline you use at inference time (i.e., wav2vec or equivalent feature extractor and the teacher model that emits target tokens).

3. **Run command**
   - Pick an output directory for checkpoints/logs.
   - Choose hyperparameters (batch size, lr, epochs, etc.).
   - Launch training, for example:
     ```bash
     python tools/train_audio_adapter.py \
       --manifest /path/to/train_manifest.json \
       --output-dir outputs/audio_adapter \
       --batch-size 2 --epochs 5 --lr 1e-4 --mixed-precision
     ```

## Dataset format

Use a JSON manifest (list of objects). Each item describes a single clip:

```json
[
  {
    "audio": "/abs/path/to/audio_embed.pt",
    "target": "/abs/path/to/teacher_tokens.pt",
    "weight": 1.0
  }
]
```

Field details:

- `audio`: Path to a `.pt` tensor produced by the same embedding extraction used at inference (`generate_infinitetalk.get_embedding`). Shape should be `[T, L, C]` where `T` = frames, `L` = wav2vec (or equivalent) layer count, `C` = hidden size.
- `target`: Path to a `.pt` tensor with the teacher conditioning tokens you want the adapter to match. Shape: `[T, context_tokens, output_dim]` (usually `context_tokens=32`, `output_dim=768`). You can collect this by running the current InfiniteTalk pipeline and saving the audio conditioning tokens it produces.
- `weight` (optional): Scalar multiplier for the example loss.

### How to generate the tensors

1. **Audio embeddings (`audio`)**: Run your existing inference-time audio feature extractor (the same one used in `generate_infinitetalk.py`) and save the full sequence of frame embeddings to disk as a float tensor.
2. **Teacher tokens (`target`)**: Run the teacher model (the released adapter or a higher-quality teacher) on the same audio so it emits per-frame conditioning tokens, then save them as a tensor.
3. Add each pair to the manifest JSON with absolute (or repo-relative) paths.

## What the loader does for you

- Builds sliding audio windows identical to inference (`wan/multitalk.py` logic) via `_build_audio_windows`.
- Pads variable-length clips inside a batch so shapes line up.
- Applies optional example weights and masks padding frames when computing the loss.

## Tips

- Keep `--audio-window` odd (default 5) so there is a center frame.
- Set `--vae-scale` to the same temporal downsample factor you use in the VAE branch (default 4).
- Start with `--mixed-precision` on GPU to speed up training.

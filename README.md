# TurnGuide

[![arXiv](https://img.shields.io/badge/arXiv-2508.07375-red)](https://arxiv.org/abs/2508.07375)
[![Hugging Face 2:1](https://img.shields.io/badge/%F0%9F%A4%97%20HF-qqjz%2Fturnguide_loss_2_1-yellow)](https://huggingface.co/qqjz/turnguide_loss_2_1)
[![Hugging Face 3:1](https://img.shields.io/badge/%F0%9F%A4%97%20HF-qqjz%2Fturnguide_loss_3_1-yellow)](https://huggingface.co/qqjz/turnguide_loss_3_1)

## News

- 🎉🎉🎉 TurnGuide has been accepted to the Interspeech 2026 Long Paper Track!

Github repository for paper: [TurnGuide: Enhancing Meaningful Full Duplex Spoken Interactions via Dynamic Turn-Level Text-Speech Interleaving](https://arxiv.org/abs/2508.07375)

Explore the [demo](https://dreamtheater123.github.io/TurnGuide-Demo/)!

This work introduces an end-to-end full-duplex speech language model and strengthens its capabilities to generate coherent, meaningful full duplex dialogues. As an initial release, we provide the test splits from the Fisher and Candor datasets to support fair and straightforward benchmarking for future research.

## Inference code

This repository includes the TurnGuide inference demo together with the GLM-4-Voice code modules needed to run it:

- `turnguide_inference.py`: the current TurnGuide inference script.
- `turnguide_inference_reproducible.py`: a legacy/reproducible inference variant.
- `flow_inference.py`, `speech_tokenizer/`, `cosyvoice/`, and `third_party/Matcha-TTS/`: supporting code from GLM-4-Voice and its decoder stack.

The model weights are not included in this repository. Download the GLM-4-Voice decoder separately and pass model paths explicitly when needed.

Model assets:

- TurnGuide checkpoint, loss ratio 2:1: [qqjz/turnguide_loss_2_1](https://huggingface.co/qqjz/turnguide_loss_2_1)
- TurnGuide checkpoint, loss ratio 3:1: [qqjz/turnguide_loss_3_1](https://huggingface.co/qqjz/turnguide_loss_3_1)
- GLM-4-Voice tokenizer: [zai-org/glm-4-voice-tokenizer](https://huggingface.co/zai-org/glm-4-voice-tokenizer)
- GLM-4-Voice decoder: [zai-org/glm-4-voice-decoder](https://huggingface.co/zai-org/glm-4-voice-decoder)

The two TurnGuide checkpoints differ only in the training loss ratio between text tokens and speech tokens:

- `qqjz/turnguide_loss_2_1`: text:speech token loss ratio = 2:1
- `qqjz/turnguide_loss_3_1`: text:speech token loss ratio = 3:1

Both checkpoints can be used with the same inference script by changing `--model-path`.

## Environment

The recommended environment is exported from the tested `torch250cu121` setup used for TurnGuide inference. It uses Python 3.10, PyTorch 2.5.0, torchaudio 2.5.0, and CUDA 12.1.

```bash
conda env create -f environment.yml
conda activate turnguide
```

The environment pins `mkl<2025` because newer MKL builds can break PyTorch 2.5.0 imports with an `iJIT_NotifyEvent` symbol error. It also includes `tiktoken` for the TurnGuide tokenizer remote code and `ruamel.yaml==0.18.6` for GLM-4-Voice decoder config loading through HyperPyYAML.

To check the core runtime versions:

```bash
python -c "import torch, torchaudio, transformers; print(torch.__version__); print(torch.version.cuda); print(torchaudio.__version__); print(transformers.__version__); print(torch.cuda.is_available())"
```

Expected core versions:

```text
torch 2.5.0
CUDA 12.1
torchaudio 2.5.0
transformers 4.44.1
```

If you already have a CUDA 12.1 PyTorch environment, you can install only the pip-side dependencies with:

```bash
pip install -r requirements.txt
```

## Run inference

```bash
git clone https://huggingface.co/zai-org/glm-4-voice-decoder

python turnguide_inference.py \
  --input-audio examples/audio/fe_03_05844_first_2min_left.wav \
  --model-path qqjz/turnguide_loss_2_1 \
  --tokenizer-path zai-org/glm-4-voice-tokenizer \
  --flow-path ./glm-4-voice-decoder \
  --output-dir ./turnguide_demo_output
```

To use the 3:1 checkpoint, replace `--model-path qqjz/turnguide_loss_2_1` with `--model-path qqjz/turnguide_loss_3_1`. You can also pass a local checkpoint directory.

This repository includes `examples/audio/fe_03_05844_first_2min_left.wav` as a small example input for the inference command above. You can replace it with any mono user-audio WAV file.

## Data

Examples:

- Fisher: `fe_03_11632_60_180.wav`
  - ID: `fe_03_11632`
  - Time segment: 60-180s
- Candor: `46f8e9b8-f80a-48cf-90a0-2e29908202c0_420.0_540.0.wav`
  - ID: `46f8e9b8-f80a-48cf-90a0-2e29908202c0`
  - Time segment: 420-540s

Download [Fisher](https://catalog.ldc.upenn.edu/LDC2004S13) and [Candor](https://guscooney.com/candor-dataset/) datasets.

## Acknowledgements and license

This code release builds on [GLM-4-Voice](https://github.com/THUDM/GLM-4-Voice). The included GLM-4-Voice code is licensed under Apache-2.0. GLM-4-Voice model weights are governed by their own model license and must be downloaded separately.

## Citation

```bibtex
@article{turnguide2026,
  title={TurnGuide: Enhancing Meaningful Full Duplex Spoken Interactions via Dynamic Turn-Level Text-Speech Interleaving},
  author={Cui, Wenqian and Zhu, Lei and Li, Xiao-Hui and Guo, Zhihan and Bai, Haoli and Hou, Lu and King, Irwin},
  journal={arXiv preprint arXiv:2508.07375},
  year={2026}
}
```

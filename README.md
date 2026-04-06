# turboquant
Standalone TurboQuant KV Cache Inference for https://huggingface.co/g023/Qwen3-1.77B-g023

~~~
run_tquant.py — Standalone TurboQuant KV Cache Inference for https://huggingface.co/g023/Qwen3-1.77B-g023
Author: g023 (https://github.com/g023) (https://huggingface.co/g023) 

Implements TurboQuant (ICLR 2026, arXiv:2504.19874) KV cache compression
directly inside a Transformers inference script. All algorithms are self-contained. Minimal dependencies. 

Algorithms:
  1. Random orthogonal rotation (QR decomposition) → Beta-distributed coordinates
  2. Lloyd-Max optimal scalar quantization → MSE-optimal centroids
  3. QJL sign-bit residual correction → unbiased inner products
  4. Group quantization for values → per-group min-max

Model Repo: [ https://huggingface.co/g023/Qwen3-1.77B-g023 ]
Model Info: (head_dim=128, 8 KV heads, 29 layers, GQA)
Model Instructions: Download files from MODEL REPO and throw in ./Qwen3-BEST and then run this program. 

Reqs:
pip install transformers datasets scipy
~~~

# instructions:
1) Download the model files and throw in ./Qwen3-BEST
2) Install pre-reqs: `pip install transformers datasets scipy`
3) Run `python run_tquant.py` for single-shot inference
4) Run `python run_tquant.py -i` or `python run_tquant.py --interactive` for multi-turn chat

# requirements:
- Python 3.10+
- transformers >= 5.0.0 (uses new layer-based Cache API)
- scipy
- CUDA GPU (tested on RTX 3060 12GB)

# features:
- TurboQuant KV cache compression (~1.55x memory savings)
- 4-bit key quantization (3-bit MSE + 1-bit QJL residual)
- 4-bit value quantization (group min-max)
- Qwen3 thinking mode support (automatic `<think>` tag parsing)
- Multi-turn interactive chat mode (`-i` / `--interactive`)
- Streaming output with real-time token display
- Self-test on startup validates quantization math
- Accurate token counting from model output
- Pre-computed Lloyd-Max codebooks (cached in ./codebooks/)

# generation parameters:
Uses model-recommended settings from generation_config.json:
- temperature: 0.6
- top_k: 20
- top_p: 0.95
- repetition_penalty: 1.1

# results (RTX 3060, single-shot):
~~~
------------------------------------------------------------
Results:
  Total tokens: ~1178
  Time: ~26s
  Tokens/sec: ~45

  TurboQuant Memory Report:
    Sequence length:    1225
    Compressed tokens:  771
    Buffer tokens:      454
    Compressed layers:  24
    Full prec. layers:  5
    Actual KV cache:    89.64 MB
    Full precision:     138.77 MB
    Compression ratio:  1.55x
    Savings:            49.13 MB
------------------------------------------------------------
~~~

# multi-turn example:
~~~
$ python run_tquant.py -i
TurboQuant Interactive Chat (type 'quit' or 'exit' to stop)

You: What is 2+2?
Assistant: 2 + 2 = 4.
  [110 tokens, 2.1s, 52.4 tok/s]

You: Multiply that by 3
Assistant: 4 × 3 = 12.
  [109 tokens, 1.8s, 60.6 tok/s]

You: quit
[Exiting]
~~~

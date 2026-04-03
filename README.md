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
2) Install pre-reqs
3) Run run_tquant.py to see it in action


# results:
~~~
------------------------------------------------------------
Results:
  Total tokens: 1509
  Time: 26.23s
  Tokens/sec: 57.53

  TurboQuant Memory Report:
    Sequence length:    1223
    Compressed tokens:  771
    Buffer tokens:      452
    Compressed layers:  24
    Full prec. layers:  5
    Actual KV cache:    89.41 MB
    Full precision:     138.54 MB
    Compression ratio:  1.55x
    Savings:            49.13 MB
------------------------------------------------------------
~~~

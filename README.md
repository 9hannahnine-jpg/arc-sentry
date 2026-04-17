# Arc Sentry v3.0.0

**Pre-generation prompt injection detection for open source LLMs.**  
Blocks attacks before `model.generate()` is called.

[![PyPI](https://img.shields.io/pypi/v/arc-sentry)](https://pypi.org/project/arc-sentry/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/9hannahnine-jpg/arc-sentry/blob/main/arc_sentry_quickstart.ipynb)


---

## Benchmark — v3.0.0

| Metric | Result |
|---|---|
| Detection rate | **100%** |
| False positive rate | **0%** |
| Session requests | 450 |
| Latency | 42ms/req |
| Layer SNR (Mistral 7B) | 2.053 |
| FR separation | 0.0787 |

450-request session benchmark on Mistral-7B-Instruct-v0.2.  
270 normal requests, 180 injection attempts (dense + subtle roleplay/hypothetical).  
Zero false positives across all safe blocks.

Also validated: Garak promptinject suite 192/192 blocked, Crescendo flagged Turn 3 (LLM Guard: 0/8).

---

## Install

```bash
pip install arc-sentry
```

## Usage

```python
from arc_sentry import ArcSentryV3, MistralAdapter
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    torch_dtype=torch.float16, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2')

adapter = MistralAdapter(model, tokenizer)
sentry = ArcSentryV3(adapter, route_id="my-deployment")
sentry.calibrate(warmup_prompts)  # ~100 prompts from your deployment

response, result = sentry.observe_and_block(user_prompt)
if result["blocked"]:
    pass  # model.generate() was never called
```

## How it works

Three detection layers:

1. **Phrase check** — 80+ injection patterns, zero latency
2. **Geometric detection** — mean-pooled hidden states at optimal layer, Fisher-Rao distance from calibrated centroid. Catches injections with no explicit language.
3. **Session D(t) monitor** — stability scalar over rolling request history. Catches gradual campaigns (Crescendo-style) invisible to single-request detection.

Grounded in the second-order Fisher manifold (H2 x H2, R = -4, tau* = sqrt(3/2) ~= 1.2247).  
Full theory: [bendexgeometry.com/theory](https://bendexgeometry.com/theory)

## Detection mechanism

```
1. Mean-pool hidden states at layer L (validated: L=16 on Mistral-7B)
2. L2-normalize: h = h / ||h||
3. Fisher-Rao distance to warmup centroid
4. Distance > threshold -> BLOCK (phrase check runs in parallel)
   model.generate() is never called
```

## Also available

- **Arc Vigil** — training stability monitor. 100% detection, 0% FP, 90% auto-recovery.  
  `pip install arc-vigil`

---

**Bendex Geometry LLC · Patent Pending · 2026 Hannah Nine**  
[bendexgeometry.com](https://bendexgeometry.com) · [PyPI](https://pypi.org/project/arc-sentry/)
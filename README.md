# Bendex Arc Sentry

White-box behavioral monitor for open source LLMs. Detects and blocks prompt injection and behavioral drift **before the model generates a response**, using residual stream layer deltas.

## What it does

Arc Sentry hooks into the residual stream of transformer models and scores the model's internal decision state before calling generate(). Anomalous inputs are blocked before a single token is produced.

No proxy monitor can do this. Helicone, Arize, and every output-based monitor see responses after they are generated. Arc Sentry sees the residual stream before generation.

## Validated results

| Model | Architecture | FP rate | Injection | Verbosity drift | Refusal drift | Trials |
|-------|-------------|---------|-----------|-----------------|---------------|--------|
| Mistral 7B | Mistral | 0% | 100% | 100% | 100% | 5/5 |
| Qwen 2.5 7B | Qwen | 0% | 100% | 100% | 100% | 5/5 |

Detection happens before model.generate() is called. Warmup required: 5 requests.

## Key finding

Different behavior types encode at different depths in the residual stream:

- Injection (control hijack): encodes at ~93% depth
- Verbosity drift (style/format): encodes at ~64% depth
- Refusal drift (policy shift): encodes at ~93% depth

Auto-layer selection finds the right layers per model automatically during calibration.

## Install

    pip install bendex

## Usage

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from bendex.whitebox import ArcSentry
    import torch

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

    sentry = ArcSentry(model, tokenizer)
    sentry.calibrate(your_warmup_prompts)

    response, result = sentry.observe_and_block(user_prompt)
    if result["blocked"]:
        pass  # model.generate() was never called

## How it works

1. Extract layer delta h[L] - h[L-1] at the decision layer
2. L2-normalize the delta vector
3. Compute cosine distance to warmup centroid
4. If distance exceeds threshold, block. generate() never runs.

During calibration, Arc Sentry scans late layers and auto-selects the best layer per behavior class.

## Theoretical foundation

Built on the second-order Fisher manifold H2 x H2 with Ricci scalar R = -4. The phase transition at tau* = sqrt(3/2) = 1.2247 (Landauer threshold) grounds the geometric interpretation of behavioral drift.

Blind predictions from the framework:
- alphas(MZ) = 0.1171 vs PDG 0.1179 +/- 0.0010 (0.8 sigma, no fitting)
- Fine structure constant to 8 significant figures from manifold curvature

Papers: bendexgeometry.com

## Honest constraints

Works best on single-domain deployments. The warmup baseline should reflect your deployment's normal traffic. Cross-domain universal detection requires larger warmup or domain routing.

## Proxy Sentry (API-based models)

    import bendex
    bendex.monitor(deployment="my-app", model="gpt-4o")

Dashboard: web-production-6e47f.up.railway.app/dashboard

## License

Bendex Source Available License. Patent Pending.
2026 Hannah Nine / Bendex Geometry LLC

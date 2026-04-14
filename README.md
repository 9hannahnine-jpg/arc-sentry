# Bendex Arc Sentry

White-box pre-generation behavioral guardrail for open source LLMs.

Arc Sentry hooks into the residual stream and detects anomalous inputs before the model generates a response. If flagged, generate() is never called.

This is different from standard monitoring tools, which operate on outputs, latency, or API-level signals.

## Validated results

| Model | Architecture | FP rate | Injection | Verbosity drift | Refusal drift | Trials |
|-------|-------------|---------|-----------|-----------------|---------------|--------|
| Mistral 7B | Mistral | 0% | 100% | 100% | 100% | 5/5 |
| Qwen 2.5 7B | Qwen | 0% | 100% | 100% | 100% | 5/5 |
| Llama 3.1 8B | Llama | 0% | 100% | 100% | 100% | 5/5 |

Zero variance across all trials. Detection happens before model.generate() is called.

## Core mechanism

1. Extract residual stream transition: delta_h = h[L] - h[L-1]
2. L2-normalize: delta_h_hat = delta_h / norm(delta_h)
3. Compute cosine distance to warmup centroid
4. If distance exceeds threshold -- block. generate() never runs.

## Key finding

Behavioral modes are encoded as layer-localized residual transitions, not uniformly across the network.

Different behaviors localize at different depths:
- Injection (control hijack): ~93% depth
- Refusal drift (policy shift): ~93% depth
- Verbosity drift (style/format): ~64% depth

Arc Sentry automatically identifies the most informative layers per model during calibration. Warmup required: 5 requests, no labeled data.

## Install

    pip install bendex

    # whitebox dependencies
    pip install bendex[whitebox]

## Usage

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from bendex.whitebox import ArcSentry
    import torch

    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

    sentry = ArcSentry(model, tokenizer)
    sentry.calibrate(warmup_prompts)

    response, result = sentry.observe_and_block(user_prompt)
    if result["blocked"]:
        pass  # model.generate() was never called

## Arc Sentry v2 (modular)

    from arc_sentry_v2.core.pipeline import ArcSentryV2
    from arc_sentry_v2.models.qwen_adapter import QwenAdapter  # or LlamaAdapter, MistralAdapter

    adapter = QwenAdapter(model, tokenizer)
    sentry = ArcSentryV2(adapter, route_id="customer-support")
    sentry.calibrate(warmup_prompts, probe_injection=injection_probes)
    response, result = sentry.observe_and_block(prompt)

## Honest constraints

Works best on single-domain deployments -- customer support bots, enterprise copilots, internal tools, fixed-use-case APIs. The warmup baseline should reflect your deployment's normal traffic. Cross-domain universal detection requires larger warmup or domain routing.

## Theoretical foundation

Built on the second-order Fisher manifold H2 x H2 with Ricci scalar R = -4. The phase transition at tau* = sqrt(3/2) = 1.2247 (Landauer threshold) grounds the geometric interpretation of behavioral drift.

Blind predictions from the framework:
- alphas(MZ) = 0.1171 vs PDG 0.1179 +/- 0.0010 (0.8 sigma, no fitting)
- Fine structure constant to 8 significant figures from manifold curvature

Papers: bendexgeometry.com

## Proxy Sentry (API-based models)

For closed-source models, the proxy-based Arc Sentry routes requests through a monitoring layer.

Dashboard: web-production-6e47f.up.railway.app/dashboard

## License

Bendex Source Available License. Patent Pending.
2026 Hannah Nine / Bendex Geometry LLC

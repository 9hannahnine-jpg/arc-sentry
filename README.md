# Bendex Arc Sentry

White-box pre-generation behavioral guardrail for open source LLMs.

Arc Sentry hooks into the residual stream and detects anomalous inputs before the model generates a response. If flagged, generate() is never called.

This is different from standard monitoring tools, which operate on outputs, latency, or API-level signals.

## Validated results

| Model | Architecture | FP rate | Detection | Prompts | Date |
|-------|-------------|---------|-----------|---------|------|
| Mistral 7B Instruct v0.2 | Mistral | 0% | 100% | 195 | April 2026 |
| Qwen 2.5 7B Instruct | Qwen | 0% | 100% | 195 | April 2026 |
| Llama 3.1 8B Instruct | Llama | 0% | 100% | 195 | April 2026 |

Zero false positives. Zero missed injections. Across three architectures, 585 total prompts.
Detection happens before model.generate() is called.

### Benchmark structure

Two-session benchmark per model:
- Session 1: 80 normal prompts (customer support, general knowledge, technical support, medical/legal/finance)
- Session 2: 115 injection prompts (10 attack categories: direct, indirect, persona hijack, jailbreak classics, social engineering, instruction injection via content, authority claims, philosophical manipulation, multi-turn style, encoding/obfuscation, gaslighting)

### Detection layers

1. **Phrase detection** — architecture-agnostic, zero latency, catches explicit injection language
2. **Fisher-Rao geometric detection** — residual stream delta at best layer vs warmup centroid, catches injections with no explicit language
3. **Session D(t) monitoring** — stability scalar (Nine 2026b) over rolling request history, catches gradual injection campaigns invisible to single-request detection

## Core mechanism

1. Extract residual stream transition: Δh = h[L] − h[L-1]
2. L2-normalize: Δh_hat = Δh / ‖Δh‖
3. Compute Fisher-Rao geodesic distance to warmup centroid: d(u,v) = arccos(u·v)
4. Threshold set from probe separation during calibration
5. If distance exceeds threshold — block. generate() never runs.

Fisher-Rao geodesic distance is used throughout — not cosine distance. This is the geometrically correct metric on the unit hypersphere and is consistent with the theoretical framework grounding the noise floor at τ* = √(3/2).

## Key finding

Behavioral modes are encoded as layer-localized residual transitions, not uniformly across the network.

Different behaviors localize at different depths:
- Injection (control hijack): ~93% depth
- Refusal drift (policy shift): ~93% depth
- Verbosity drift (style/format): ~64% depth

Arc Sentry automatically identifies the most informative layers per model during calibration. Warmup required: 10 prompts, no labeled data.

## Install

    pip install bendex

    # whitebox dependencies
    pip install bendex[whitebox]

## Usage

### v1 (single file)

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

### v2 (modular, recommended)

    from arc_sentry_v2.core.pipeline import ArcSentryV2
    from arc_sentry_v2.models.mistral_adapter import MistralAdapter  # or QwenAdapter, LlamaAdapter

    adapter = MistralAdapter(model, tokenizer)
    sentry = ArcSentryV2(adapter, route_id="customer-support")
    sentry.calibrate(warmup_prompts)
    response, result = sentry.observe_and_block(prompt)

    if result["blocked"]:
        pass  # generate() was never called
    else:
        print(result["snr"])  # signal-to-noise ratio vs τ*

## Honest constraints

Works best on single-domain deployments — customer support bots, enterprise copilots, internal tools, fixed-use-case APIs. The warmup baseline should reflect your deployment's normal traffic. Cross-domain universal detection requires larger warmup or domain routing.

## Theoretical foundation

Built on the second-order Fisher manifold H² × H² with Ricci scalar R = −4. The phase transition at τ* = √(3/2) ≈ 1.2247 (Landauer threshold) grounds the geometric interpretation of behavioral drift.

Detection uses Fisher-Rao geodesic distance — the geometrically correct metric on the unit hypersphere. The threshold is derived from probe separation during calibration, not from a tuned hyperparameter.

Blind predictions from the framework:
- αs(MZ) = 0.1171 vs PDG 0.1179 ± 0.0010 (0.8σ, no fitting)
- Fine structure constant to 8 significant figures from manifold curvature

Papers: bendexgeometry.com

## Proxy Sentry (API-based models)

For closed-source models (GPT-4, Claude, Gemini), the proxy-based Arc Sentry routes requests through a monitoring layer with no model access required.

Dashboard: web-production-6e47f.up.railway.app/dashboard

## License

Bendex Source Available License. Patent Pending.
2026 Hannah Nine / Bendex Geometry LLC

import torch
import numpy as np
import math
from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class BaseModelAdapter(ABC):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.n_layers = model.config.num_hidden_layers
        self.hidden_dim = model.config.hidden_size

    @abstractmethod
    def format_prompt(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        pass

    def mean_pooled_repr(self, text: str, layer: int) -> np.ndarray:
        """
        Mean-pool hidden states across all tokens at layer.
        Returns L2-normalized float32 vector.
        Validated: DR=100%, FPR=0% on Mistral-7B (450 requests, Nine 2026).
        """
        enc = self.tokenizer(
            text, return_tensors="pt",
            truncation=True, max_length=128
        ).to(self.model.device)
        with torch.no_grad():
            out = self.model(**enc, output_hidden_states=True)
        h = out.hidden_states[layer][0].mean(dim=0).cpu().float().numpy()
        norm = np.linalg.norm(h)
        return h / (norm + 1e-10)

    def extract_hidden(self, prompt: str, layers: List[int],
                       system_prompt: Optional[str] = None,
                       pool: str = "mean") -> Dict[int, np.ndarray]:
        full_input = self.format_prompt(prompt, system_prompt)
        inputs = self.tokenizer(full_input, return_tensors="pt").to(self.model.device)
        captured = {}
        hooks = []

        def make_hook(idx):
            def hook(module, input, output):
                hidden = output[0] if isinstance(output, tuple) else output
                if pool == "mean":
                    captured[idx] = hidden.detach().float().mean(dim=1)[0].cpu()
                elif pool == "last":
                    captured[idx] = hidden.detach().float()[:, -1, :][0].cpu()
            return hook

        for li in layers:
            hooks.append(self.model.model.layers[li].register_forward_hook(make_hook(li)))

        with torch.no_grad():
            _ = self.model(**inputs)

        for h in hooks:
            h.remove()

        return {k: v.numpy() for k, v in captured.items()}

    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                 max_new_tokens: int = 60) -> str:
        full_input = self.format_prompt(prompt, system_prompt)
        inputs = self.tokenizer(full_input, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

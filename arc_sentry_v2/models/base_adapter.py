
import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple


class BaseModelAdapter(ABC):
    """
    Wraps a HuggingFace model and handles:
    - chat template formatting
    - hook registration
    - hidden state extraction
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.n_layers = model.config.num_hidden_layers
        self.hidden_dim = model.config.hidden_size

    @abstractmethod
    def format_prompt(self, prompt: str,
                      system_prompt: Optional[str] = None) -> str:
        pass

    def extract_hidden(self, prompt: str,
                       layers: List[int],
                       system_prompt: Optional[str] = None,
                       pool: str = "mean") -> Dict[int, np.ndarray]:
        """
        Extract mean-pooled hidden states at specified layers.
        Returns dict of {layer_idx: numpy array of shape [hidden_dim]}
        """
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
            hooks.append(
                self.model.model.layers[li].register_forward_hook(make_hook(li)))

        with torch.no_grad():
            _ = self.model(**inputs)

        for h in hooks:
            h.remove()

        return {k: v.numpy() for k, v in captured.items()}

    def generate(self, prompt: str,
                 system_prompt: Optional[str] = None,
                 max_new_tokens: int = 60) -> str:
        full_input = self.format_prompt(prompt, system_prompt)
        inputs = self.tokenizer(full_input, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

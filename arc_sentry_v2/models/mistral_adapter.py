
from typing import Optional
from arc_sentry_v2.models.base_adapter import BaseModelAdapter


class MistralAdapter(BaseModelAdapter):
    def format_prompt(self, prompt: str,
                      system_prompt: Optional[str] = None) -> str:
        try:
            msgs = []
            if system_prompt:
                msgs.append({"role": "system", "content": system_prompt})
            msgs.append({"role": "user", "content": prompt})
            return self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            if system_prompt:
                return f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]"
            return f"[INST] {prompt} [/INST]"

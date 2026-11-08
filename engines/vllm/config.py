from dataclasses import dataclass
from typing import Any

@dataclass
class VLLMRunConfig:
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    max_model_len: int = 4096
    max_tokens: int = 32

    extra_engine_args: dict[str, Any] | None = None

    def engine_args(self) -> dict[str, Any]:
        args = {
            "model": self.model,
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": 0.3,
        }

        if self.extra_engine_args:
            args.update(self.extra_engine_args)

        return args
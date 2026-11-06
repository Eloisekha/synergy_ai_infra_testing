import sys

import torch
import vllm
from vllm import LLM, SamplingParams

from config import VLLMRunConfig


def print_environment():
    print("=== Environment ===")
    print(f"Python: {sys.version.split()[0]}")
    print(f"vLLM: {vllm.__version__}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")


def run(config: VLLMRunConfig):
    print_environment()

    print("\n=== Configuration ===")
    print(config)

    llm = LLM(**config.engine_args())

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=config.max_tokens,
    )

    prompts = [
        "Explain what a GPU is in one sentence."
    ]

    outputs = llm.generate(
        prompts,
        sampling_params,
    )

    print("\n=== Output ===")

    for output in outputs:
        print(output.outputs[0].text.strip())


def main():
    from flags import FLAGS
    from pairs import pair_configs

    flag_f = FLAGS["chunked_prefill"]
    flag_g = FLAGS["prefix_caching"]

    configs = pair_configs(flag_f, flag_g)

    for state, extra_args in configs.items():
        print(f"{state}: {extra_args}")


if __name__ == "__main__":
    main()
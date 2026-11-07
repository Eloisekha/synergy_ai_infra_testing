import sys
import json
import os
import traceback
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


def run(config: VLLMRunConfig, state: str):
    print_environment()

    print("\n=== Configuration ===")
    print(config)

    report = {
        "state": state,
        "config": config.engine_args(),
    }

    try:
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

        text = outputs[0].outputs[0].text.strip()

        print("\n=== Output ===")
        print(text)

        report.update({
            "status": "ok",
            "output": text,
            "error_type": None,
            "error": None,
        })

        success = True

    except Exception as exc:
        print("\n=== Error ===")
        print(f"{type(exc).__name__}: {exc}")

        report.update({
            "status": "error",
            "output": None,
            "error_type": type(exc).__name__,
            "error": str(exc),
        })

        traceback.print_exc()

        success = False

    os.makedirs("reports", exist_ok=True)

    with open(f"reports/{state}.json", "w") as f:
        json.dump(report, f, indent=2)

    return success


def main():
    from flags import FLAGS
    from pairs import pair_configs

    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python engines/vllm/run.py <00|10|01|11>"
        )

    state = sys.argv[1]

    configs = pair_configs(
        FLAGS["chunked_prefill"],
        FLAGS["prefix_caching"],
    )

    if state not in configs:
        raise SystemExit(
            f"Invalid state: {state}. Choose one of: 00, 10, 01, 11"
        )

    config = VLLMRunConfig(
        extra_engine_args=configs[state]
    )

    print(f"=== State: {state} ===")
    
    success = run(config, state)

    if not success:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
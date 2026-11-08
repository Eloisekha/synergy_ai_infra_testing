import sys
import json
import os
import traceback
import torch
import vllm
from vllm import LLM, SamplingParams

from config import VLLMRunConfig
from flags import FLAGS
from pairs import pair_configs, report_dir, DEFAULT_FLAG_F, DEFAULT_FLAG_G

def print_environment():
    print("=== Environment ===")
    print(f"Python: {sys.version.split()[0]}")
    print(f"vLLM: {vllm.__version__}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")


def run(config: VLLMRunConfig, state: str, flag_f:str, flag_g:str):
    print_environment()

    old_report = os.path.join(report_dir(flag_f, flag_g), f"{state}.json")
    if os.path.exists(old_report):
        os.remove(old_report)

    print("\n=== Configuration ===")
    print(config)

    report = {
        "state": state,
        "flag_f": flag_f,
        "flag_g": flag_g,
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

    out_dir = report_dir(flag_f, flag_g)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, f"{state}.json"), "w") as f:
        json.dump(report, f, indent=2)

    return success


def main():

    if len(sys.argv) not in (2, 4):
        raise SystemExit(
            "Usage: python engines/vllm/run.py <00|10|01|11> [flag_f flag_g]"
        )

    state = sys.argv[1]
    flag_f = sys.argv[2] if len(sys.argv) == 4 else DEFAULT_FLAG_F
    flag_g = sys.argv[3] if len(sys.argv) == 4 else DEFAULT_FLAG_G

    for name in (flag_f, flag_g):
        if name not in FLAGS:
            raise SystemExit(
                f"Unknown flag: {name}. Choose one of: {', '.join(FLAGS)}"
            )

    if flag_f == flag_g:
        raise SystemExit("flag_f and flag_g must be different flags")

    configs = pair_configs(FLAGS[flag_f], FLAGS[flag_g])

    if state not in configs:
        raise SystemExit(
            f"Invalid state: {state}. Choose one of: 00, 10, 01, 11"
        )

    config = VLLMRunConfig(
        extra_engine_args=configs[state]
    )

    print(f"=== State: {state} ===")
    
    success = run(config, state, flag_f, flag_g)

    if not success:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
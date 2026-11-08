import sys
import json
import os
import traceback
import torch
import vllm
from vllm import LLM, SamplingParams

from config import VLLMRunConfig
from flags import FLAGS
from pairs import STATES, pair_configs, report_dir, resolve_pair

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

    out_dir = report_dir(flag_f, flag_g)
    report_path = os.path.join(out_dir, f"{state}.json")

    # Remove a stale report before starting a new run.
    if os.path.exists(report_path):
        os.remove(report_path)

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

        #### Original prompt, commented out (kept for reference) ####
        # prompts = [
        #     "Explain what a GPU is in one sentence."
        # ]

        #### to read the 20 test prompts from prompts.json ####
        prompts_path = os.path.join(os.path.dirname(__file__), "prompts.json")
        with open(prompts_path) as f:
            tests = json.load(f)
        prompts = [t["prompt"] for t in tests]

        outputs = llm.generate(
            prompts,
            sampling_params,
        )

        text = outputs[0].outputs[0].text.strip()

        #### to collect all 20 answer ####
        answers = [o.outputs[0].text.strip() for o in outputs]
        text = "\n".join(answers)

        print("\n=== Output ===")
        print(text)

        report.update({
            "status": "ok",
            "output": text,
            "error_type": None,
            "error": None,
        })

        #### to save each prompt's correct answer next to vLLM's answer ####
        report["results"] = [
            {"id": t["id"], "correct_answer": t["answer"], "vllm_answer": a}
            for t, a in zip(tests, answers)
        ]

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

    os.makedirs(out_dir, exist_ok=True)

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    return success


def main():
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: python engines/vllm/run.py "
            "<00|10|01|11> [flag_f flag_g]"
        )

    state = sys.argv[1]

    if state not in STATES:
        raise SystemExit(
            f"Invalid state: {state}. "
            f"Choose one of: {', '.join(STATES)}"
        )

    try:
        flag_f, flag_g = resolve_pair(sys.argv[2:])
    except ValueError as exc:
        raise SystemExit(
            "Usage: python engines/vllm/run.py "
            "<00|10|01|11> [flag_f flag_g]\n"
            f"{exc}"
        )

    configs = pair_configs(
        FLAGS[flag_f],
        FLAGS[flag_g],
    )

    config = VLLMRunConfig(
        extra_engine_args=configs[state]
    )

    print(f"=== State: {state} ===")
    print(f"=== Pair: {flag_f} x {flag_g} ===")

    success = run(
        config,
        state,
        flag_f,
        flag_g,
    )

    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
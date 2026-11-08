import subprocess
import sys

from pairs import STATES, resolve_pair

def run_state(state:str, flag_f:str, flag_g:str):
    print(f"\n=== Running state {state} ===")

    result = subprocess.run(
        [sys.executable, "engines/vllm/run.py", state, flag_f, flag_g],
        check=False,
    )

    print(f"State {state} exit code: {result.returncode}")

    return result.returncode

def main():
    try:
        flag_f, flag_g = resolve_pair(sys.argv[1:])
    except ValueError as exc:
        raise SystemExit(
            "Usage: python engines/vllm/test_pair.py [flag_f flag_g]\n"
            f"{exc}"
        )

    print(f"=== Testing pair: {flag_f} x {flag_g} ===")

    for state in STATES:
        run_state(state, flag_f, flag_g)

    print("\n=== Running oracle ===")

    subprocess.run(
        [sys.executable, "engines/vllm/oracle.py", flag_f, flag_g],
        check=False,
    )

if __name__ == "__main__":
    main()
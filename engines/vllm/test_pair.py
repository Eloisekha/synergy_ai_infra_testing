import subprocess
import sys

STATES = ["00", "10", "01", "11"]

def run_state(state:str):
    print(f"\n=== Running state {state} ===")

    result = subprocess.run(
        [sys.executable, "engines/vllm/run.py", state],
        check=False,
    )

    print(f"State {state} exit code: {result.returncode}")

def main():
    for state in STATES:
        run_state(state)

    print("\n=== Running oracle ===")

    subprocess.run(
        [sys.executable, "engines/vllm/oracle.py"],
        check=False,
    )

if __name__ == "__main__":
    main()
import json

STATES = ["00", "01", "10", "11"]

def load(state):
    with open(f"reports/{state}.json") as f:
        return json.load(f)

def check(report, baseline):
    started_ok = report["status"] == "ok"
    output_not_empty = bool((report.get("output") or "").strip())

    return {
        "started_ok": started_ok,
        "output_not_empty": output_not_empty,
        "output_changed": report.get("output") != baseline.get("output"),
        "verdict": "pass" if started_ok and output_not_empty else "fail",
    }

def main():
    baseline = load("00")

    print("=== Differential Results ===")

    for state in STATES:
        result = check(load(state), baseline)

        print(f"\nState {state}")
        print(f"started_ok: {result['started_ok']}")
        print(f"output_not_empty: {result['output_not_empty']}")
        print(f"output_changed: {result['output_changed']}")
        print(f"verdict: {result['verdict']}")


if __name__ == "__main__":
    main()
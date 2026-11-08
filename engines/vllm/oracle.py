import json
import os
import sys

from pairs import STATES, report_dir, resolve_pair


def load(state, flag_f, flag_g):
    path = os.path.join(
        report_dir(flag_f, flag_g),
        f"{state}.json",
    )

    if not os.path.exists(path):
        return {
            "state": state,
            "flag_f": flag_f,
            "flag_g": flag_g,
            "status": "error",
            "output": None,
            "error_type": "MissingReport",
            "error": f"no report file at {path}",
        }

    with open(path) as f:
        return json.load(f)


def check(report, baseline):
    started_ok = report["status"] == "ok"
    output_not_empty = bool(
        (report.get("output") or "").strip()
    )

    return {
        "started_ok": started_ok,
        "output_not_empty": output_not_empty,
        "output_changed": (
            report.get("output")
            != baseline.get("output")
        ),
        "verdict": (
            "pass"
            if started_ok and output_not_empty
            else "fail"
        ),
    }


def _differs(a, b):
    return a != b


def _interaction(
    x,
    flag_f_only,
    flag_g_only,
    both,
    base,
):
    """
    Check whether each flag's observed effect depends on
    the state of the other flag.
    """

    # Effect of F when G is unchanged:
    # 10 vs 00
    f_without_g = _differs(
        x[flag_f_only],
        x[base],
    )

    # Effect of F when G is already changed:
    # 11 vs 01
    f_with_g = _differs(
        x[both],
        x[flag_g_only],
    )

    # Effect of G when F is unchanged:
    # 01 vs 00
    g_without_f = _differs(
        x[flag_g_only],
        x[base],
    )

    # Effect of G when F is already changed:
    # 11 vs 10
    g_with_f = _differs(
        x[both],
        x[flag_f_only],
    )

    interaction = (
        f_without_g != f_with_g
        or g_without_f != g_with_f
    )

    return {
        "f_effect_without_g": f_without_g,
        "f_effect_with_g": f_with_g,
        "g_effect_without_f": g_without_f,
        "g_effect_with_f": g_with_f,
        "interaction": interaction,
    }


def pair_check(reports):
    missing_reports = [
        state
        for state in STATES
        if reports[state].get("error_type")
        == "MissingReport"
    ]

    if missing_reports:
        return {
            "status_effects": None,
            "status_interaction": None,
            "output_effects": None,
            "output_interaction": None,
            "interaction": None,
            "reason": (
                "pair evaluation unavailable: "
                "missing reports for "
                + ", ".join(missing_reports)
            ),
        }

    # 1. Status / error behavior

    status = {
        state: (
            reports[state]["status"],
            reports[state].get("error_type"),
        )
        for state in STATES
    }

    status_part = _interaction(
        status,
        flag_f_only="10",
        flag_g_only="01",
        both="11",
        base="00",
    )

    result = {
        "status_effects": {
            key: value
            for key, value in status_part.items()
            if key != "interaction"
        },
        "status_interaction": (
            status_part["interaction"]
        ),
    }

    # 2. Output behavior

    all_ok = all(
        reports[state]["status"] == "ok"
        for state in STATES
    )

    if all_ok:
        outputs = {
            state: reports[state].get("output")
            for state in STATES
        }

        output_part = _interaction(
            outputs,
            flag_f_only="10",
            flag_g_only="01",
            both="11",
            base="00",
        )

        result["output_effects"] = {
            key: value
            for key, value in output_part.items()
            if key != "interaction"
        }

        result["output_interaction"] = (
            output_part["interaction"]
        )

    else:
        result["output_effects"] = None
        result["output_interaction"] = None
        result["reason"] = (
            "output comparison unavailable: "
            "at least one state failed"
        )

    # 3. Overall interaction verdict

    if result["status_interaction"]:
        result["interaction"] = True

    elif result["output_interaction"] is None:
        result["interaction"] = None

    else:
        result["interaction"] = (
            result["output_interaction"]
        )

    return result


def main():
    try:
        flag_f, flag_g = resolve_pair(
            sys.argv[1:]
        )
    except ValueError as exc:
        raise SystemExit(
            "Usage: python engines/vllm/oracle.py "
            "[flag_f flag_g]\n"
            f"{exc}"
        )

    reports = {
        state: load(
            state,
            flag_f,
            flag_g,
        )
        for state in STATES
    }

    baseline = reports["00"]

    print(f"Pair: {flag_f} x {flag_g}")
    print("=== Differential Results ===")

    for state in STATES:
        result = check(
            reports[state],
            baseline,
        )

        print(f"\nState {state}")
        print(
            f"started_ok: "
            f"{result['started_ok']}"
        )
        print(
            f"output_not_empty: "
            f"{result['output_not_empty']}"
        )
        print(
            f"output_changed: "
            f"{result['output_changed']}"
        )
        print(
            f"verdict: "
            f"{result['verdict']}"
        )

    print("\n=== Pair Interaction ===")

    result = pair_check(reports)

    print("\nStatus effects:")

    if result["status_effects"] is not None:
        for key, value in (
            result["status_effects"].items()
        ):
            print(f"{key}: {value}")

        print(
            f"status_interaction: "
            f"{result['status_interaction']}"
        )
    else:
        print("unavailable")

    print("\nOutput effects:")

    if result["output_effects"] is not None:
        for key, value in (
            result["output_effects"].items()
        ):
            print(f"{key}: {value}")

        print(
            f"output_interaction: "
            f"{result['output_interaction']}"
        )
    else:
        print("unavailable")

    if result.get("reason"):
        print(
            f"reason: {result['reason']}"
        )

    print("\nOverall:")
    print(
        f"interaction: "
        f"{result['interaction']}"
    )


if __name__ == "__main__":
    main()
from oracle import pair_check


STATES = ["00", "10", "01", "11"]


def make(outputs):
    return {
        state: {
            "status": "ok",
            "output": output,
            "error_type": None,
        }
        for state, output in zip(STATES, outputs)
    }


# --------------------------------------------------
# 1. No observable effect anywhere
# --------------------------------------------------

result = pair_check(
    make(["A", "A", "A", "A"])
)

assert result["output_interaction"] is False
assert result["interaction"] is False


# --------------------------------------------------
# 2. F only has an effect when G is changed
#
# 00 = A
# 10 = A     F alone does nothing
# 01 = A
# 11 = B     F now changes output
# --------------------------------------------------

result = pair_check(
    make(["A", "A", "A", "B"])
)

assert result["output_interaction"] is True
assert result["interaction"] is True


# --------------------------------------------------
# 3. F has an effect without G,
#    but loses that effect when G is changed
#
# 00 = A
# 10 = B
# 01 = B
# 11 = B
# --------------------------------------------------

result = pair_check(
    make(["A", "B", "B", "B"])
)

assert result["output_interaction"] is True
assert result["interaction"] is True


# --------------------------------------------------
# 4. One state fails in an interaction-like pattern
#
# 00 = ok
# 10 = ok
# 01 = ok
# 11 = error
# --------------------------------------------------

broken = make(["A", "A", "A", "A"])

broken["11"]["status"] = "error"
broken["11"]["error_type"] = "RuntimeError"

result = pair_check(broken)

assert result["status_interaction"] is True
assert result["output_interaction"] is None
assert result["interaction"] is True


# --------------------------------------------------
# 5. Failure exists, but output interaction
#    cannot be evaluated.
#
# Here 00 and 10 both fail in the same way,
# while 01 and 11 both succeed.
# The status effect of F is therefore the same
# in both G contexts.
# --------------------------------------------------

broken = make(["A", "A", "A", "A"])

broken["00"]["status"] = "error"
broken["00"]["error_type"] = "RuntimeError"

broken["10"]["status"] = "error"
broken["10"]["error_type"] = "RuntimeError"

result = pair_check(broken)

assert result["status_interaction"] is False
assert result["output_interaction"] is None
assert result["interaction"] is None


print("ok")
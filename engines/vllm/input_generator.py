
def generate_inputs():
    short = [
        "Explain what a GPU is."
    ]

    long = [
        "Explain how GPUs are used in AI. " * 100
    ]

    prefix = "You are a computer science assistant. " * 20

    repeated_prefix = [
        prefix + "Explain what a GPU is.",
        prefix + "Explain what a CPU is."
    ]

    return {
        "short": short,
        "long": long,
        "repeated_prefix": repeated_prefix
    }

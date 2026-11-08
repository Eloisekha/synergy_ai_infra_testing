import os

from flags import Flag, FLAGS

STATES = ["00", "10", "01", "11"]

DEFAULT_FLAG_F = "chunked_prefill"
DEFAULT_FLAG_G = "prefix_caching"

def pair_configs(flag_f: Flag, flag_g: Flag):
    return {
        "00": {},
        "10": {
            flag_f.name: flag_f.active,
        },
        "01": {
            flag_g.name: flag_g.active,
        },
        "11": {
            flag_f.name: flag_f.active,
            flag_g.name: flag_g.active,
        },
    }

def report_dir(flag_f:str, flag_g:str) -> str:
    return os.path.join("reports", f"{flag_f}__{flag_g}")


def resolve_pair(args):
    if len(args) == 0:
        flag_f = DEFAULT_FLAG_F
        flag_g = DEFAULT_FLAG_G

    elif len(args) == 2:
        flag_f, flag_g = args

    else:
        raise ValueError(
            "expected either no flag arguments or exactly two flags"
        )

    for name in (flag_f, flag_g):
        if name not in FLAGS:
            raise ValueError(
                f"Unknown flag: {name}. "
                f"Choose one of: {', '.join(FLAGS)}"
            )

    if flag_f == flag_g:
        raise ValueError(
            "flag_f and flag_g must be different flags"
        )

    return flag_f, flag_g
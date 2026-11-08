import os

from flags import Flag

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

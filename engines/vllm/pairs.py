from flags import Flag


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
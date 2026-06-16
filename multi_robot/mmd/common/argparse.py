import yaml
import re
import logging
from typing import List, Dict, Any



## ========================== Parse and Load Config ==========================. 
def load_config(config_path: str, overrides: Dict[str, Any] = None) -> dict:
    """Load the YAML config and apply any dotted-key overrides."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if overrides:
        update_nested_config(config, overrides)
    return config

def parse_overrides(extra_args: List) -> Dict[str, Any]:
    """
    Turn a list like ['--a.b=1', '--c.d', 'hello'] into {'a.b': 1, 'c.d': 'hello'}.
    """
    overrides: Dict[str, Any] = {}
    i = 0
    while i < len(extra_args):
        token:str = extra_args[i]
        if not token.startswith("--"):
            logging.error(f"Unrecognized token {token!r}")
            raise ValueError(f"Unrecognized token {token!r}")

        # split --key=val or consume next
        if "=" in token:
            key, raw_val = token[2:].split("=", 1)
        else:
            key = token[2:]
            i += 1
            if i >= len(extra_args):
                raise ValueError(f"Missing value for `{token}`")
            raw_val = extra_args[i]

        # parse YAML literal so we get ints, bools, lists, etc.
        try:
            val = yaml.safe_load(raw_val)
            # convert quoted scientific-notation strings to float
            if isinstance(val, str):
                if re.fullmatch(r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?", val):
                    val = float(val)
        except yaml.YAMLError:
            val = raw_val

        overrides[key] = val
        i += 1

    return overrides

def update_nested_config(cfg: dict, overrides: Dict[str, Any]) -> dict:
    """
    For each 'a.b.c': v in overrides, set cfg['a']['b']['c'] = v, creating dicts as needed.
    """
    for dotted_key, v in overrides.items():
        parts = dotted_key.split(".")
        d = cfg
        for i, p in enumerate(parts[:-1]):
            if p in d and isinstance(d[p], dict):
                d = d[p]
            else:
                logging.error(f"Invalid override key: \'{'.'.join(parts[:(i+1)])}\'")
                raise KeyError(f"Invalid override key: '{'.'.join(parts[:(i+1)])}'")
        if parts[-1] in d:
            d[parts[-1]] = v
        else:
            logging.error(f"Invalid override key: \'{dotted_key}\'")
            raise KeyError(f"Invalid override key: '{dotted_key}'")
    return cfg
